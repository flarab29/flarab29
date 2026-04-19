package org.apache.spark.ml.regression

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, eigSym}

import org.apache.spark.internal.Logging
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.{Matrix, Matrices, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructType}
import org.apache.spark.storage.StorageLevel

/**
 * Cox proportional hazards regression for right-censored survival data.
 *
 * Scope of this implementation:
 *   - right censoring only
 *   - no case weights
 *   - no strata
 *   - no offsets
 *   - ties handled with Breslow or Efron
 *
 * Notes:
 *   - features are centered and scaled during fitting following the same spirit as
 *     survival::coxph / coxfit6: centering by mean and scaling by mean absolute deviation.
 *   - coefficients are returned on the original feature scale.
 *   - absolute survival predictions use a Breslow-style baseline cumulative hazard estimated
 *     on the centered/scaled training linear predictor.
 */
private[regression] trait CoxPHRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasAggregationDepth {

  final val censorCol: Param[String] = new Param[String](
    this,
    "censorCol",
    "censor column name: 1 = event occurred, 0 = right-censored")

  def getCensorCol: String = $(censorCol)

  final val tiesMethod: Param[String] = new Param[String](
    this,
    "tiesMethod",
    "method for handling ties: breslow or efron",
    ParamValidators.inArray(Array("breslow", "efron")))

  def getTiesMethod: String = $(tiesMethod)

  final val rankTolerance: DoubleParam = new DoubleParam(
    this,
    "rankTolerance",
    "relative tolerance used to detect singular or near-singular information matrices",
    ParamValidators.gt(0.0))

  def getRankTolerance: Double = $(rankTolerance)

  final val maxStepHalving: IntParam = new IntParam(
    this,
    "maxStepHalving",
    "maximum consecutive step-halving attempts before aborting the Newton step",
    ParamValidators.gtEq(1))

  def getMaxStepHalving: Int = $(maxStepHalving)

  setDefault(
    censorCol -> "censor",
    tiesMethod -> "efron",
    maxIter -> 20,
    tol -> 1e-9,
    aggregationDepth -> 2,
    rankTolerance -> 1e-10,
    maxStepHalving -> 25)

  protected def validateAndTransformSchema(schema: StructType, fitting: Boolean): StructType = {
    SchemaUtils.checkColumnType(schema, $(featuresCol), new org.apache.spark.ml.linalg.VectorUDT)
    if (fitting) {
      SchemaUtils.checkNumericType(schema, $(labelCol))
      SchemaUtils.checkNumericType(schema, $(censorCol))
    }
    SchemaUtils.appendColumn(schema, $(predictionCol), DoubleType)
  }
}

object CoxPHRegression extends DefaultParamsReadable[CoxPHRegression] {

  private[regression] final val MaxSafeExp: Double = 700.0

  private[regression] final case class CoxInstance(
      time: Double,
      status: Int,
      features: Array[Double])

  private[regression] final case class TimeSufficientStats(
      totalRisk: Double,
      totalRiskX: Array[Double],
      totalRiskXX: Array[Double],
      nDead: Int,
      deadRisk: Double,
      deadRiskX: Array[Double],
      deadRiskXX: Array[Double],
      deadX: Array[Double],
      hadNumericalCapping: Boolean)

  private[regression] final case class IterationResult(
      logLik: Double,
      score: Array[Double],
      information: Array[Array[Double]],
      hadNumericalIssue: Boolean)

  private[regression] final case class LinearSolveResult(
      step: Array[Double],
      inverse: Array[Array[Double]],
      rank: Int)

  private[regression] final case class BaselineHazard(
      times: Array[Double],
      cumulativeHazard: Array[Double])
}

class CoxPHRegression(override val uid: String)
  extends Regressor[Vector, CoxPHRegression, CoxPHRegressionModel]
    with CoxPHRegressionParams with DefaultParamsWritable with Logging {

  import CoxPHRegression._

  def this() = this(Identifiable.randomUID("coxph"))

  def setCensorCol(value: String): this.type = set(censorCol, value)

  def setTiesMethod(value: String): this.type = set(tiesMethod, value)

  def setMaxIter(value: Int): this.type = set(maxIter, value)

  def setTol(value: Double): this.type = set(tol, value)

  def setAggregationDepth(value: Int): this.type = set(aggregationDepth, value)

  def setRankTolerance(value: Double): this.type = set(rankTolerance, value)

  def setMaxStepHalving(value: Int): this.type = set(maxStepHalving, value)

  override protected def train(dataset: Dataset[_]): CoxPHRegressionModel = {
    val method = if ($(tiesMethod) == "efron") 1 else 0

    val rawInstances: RDD[CoxInstance] = dataset.select(
      col($(labelCol)).cast(DoubleType),
      col($(censorCol)).cast(DoubleType),
      col($(featuresCol))
    ).rdd.map {
      case Row(time: Double, censor: Double, features: Vector) =>
        val featureArray = features.toArray
        require(time.isFinite, s"Survival time must be finite, got $time")
        require(time >= 0.0, s"Survival time must be >= 0, got $time")
        require(censor == 0.0 || censor == 1.0,
          s"${$(censorCol)} must contain only 0.0 or 1.0, got $censor")
        require(featureArray.forall(_.isFinite), "All feature values must be finite")
        CoxInstance(time, censor.toInt, featureArray)
    }.setName("coxph_training_instances").persist(StorageLevel.MEMORY_AND_DISK)

    val nObs = rawInstances.count()
    require(nObs > 0, "CoxPHRegression requires at least one observation")

    val first = rawInstances.first()
    val nvar = first.features.length
    require(nvar > 0, "CoxPHRegression requires at least one feature")

    val inconsistentSize = rawInstances.filter(_.features.length != nvar).take(1)
    require(inconsistentSize.isEmpty,
      s"All feature vectors must have the same size ($nvar expected)")

    val nEvents = rawInstances.filter(_.status == 1).count()
    require(nEvents > 0, "CoxPHRegression requires at least one observed event")

    logInfo(s"CoxPH: fitting on $nObs observations, $nEvents events, $nvar features")

    val (meansArr, scaleArr) = computeMeansAndScale(rawInstances, nvar, nObs)
    val bcMeans = rawInstances.sparkContext.broadcast(meansArr)
    val bcScale = rawInstances.sparkContext.broadcast(scaleArr)

    val scaledInstances: RDD[CoxInstance] = rawInstances.map { inst =>
      val means = bcMeans.value
      val scales = bcScale.value
      val scaled = new Array[Double](nvar)
      var j = 0
      while (j < nvar) {
        scaled(j) = (inst.features(j) - means(j)) * scales(j)
        j += 1
      }
      CoxInstance(inst.time, inst.status, scaled)
    }.setName("coxph_scaled_training_instances")
      .persist(StorageLevel.MEMORY_AND_DISK)

    scaledInstances.count()
    rawInstances.unpersist()
    bcMeans.destroy()
    bcScale.destroy()

    val beta = Array.fill[Double](nvar)(0.0)
    val initialFit = distributedIteration(scaledInstances, beta, method)
    var currentFit = initialFit
    require(currentFit.logLik.isFinite,
      "Initial CoxPH log-likelihood is not finite; check data scaling or feature magnitudes")

    val initialSolve = solveSymmetric(currentFit.information, currentFit.score, $(rankTolerance))
    val scoreTest = quadraticForm(currentFit.score, initialSolve.step)
    val initialLogLik = currentFit.logLik

    var currentRank = initialSolve.rank
    var currentInverse = initialSolve.inverse
    var currentBeta = beta.clone()
    var bestLogLik = currentFit.logLik
    var proposal = addInPlace(currentBeta.clone(), initialSolve.step)

    var converged = false
    var iteration = 0
    var halving = 0

    while (iteration < $(maxIter) && !converged) {
      iteration += 1
      val proposedFit = distributedIteration(scaledInstances, proposal, method)
      val isFiniteIteration =
        proposedFit.logLik.isFinite &&
          !proposedFit.hadNumericalIssue &&
          allFinite(proposedFit.score) &&
          allFinite(proposedFit.information)

      if (isFiniteIteration && relativeLogLikChange(bestLogLik, proposedFit.logLik) <= $(tol)) {
        val solve = solveSymmetric(proposedFit.information, proposedFit.score, $(rankTolerance))
        currentFit = proposedFit
        currentInverse = solve.inverse
        currentRank = solve.rank
        currentBeta = proposal.clone()
        bestLogLik = proposedFit.logLik
        converged = true
      } else if (!isFiniteIteration || proposedFit.logLik < bestLogLik) {
        halving += 1
        if (halving > $(maxStepHalving)) {
          logWarning(
            s"CoxPH: reached max step-halving count ${$(maxStepHalving)} at iteration $iteration")
          iteration = $(maxIter)
        } else {
          var j = 0
          while (j < nvar) {
            proposal(j) = (proposal(j) + halving * currentBeta(j)) / (halving + 1.0)
            j += 1
          }
        }
      } else {
        val solve = solveSymmetric(proposedFit.information, proposedFit.score, $(rankTolerance))
        halving = 0
        bestLogLik = proposedFit.logLik
        currentFit = proposedFit
        currentInverse = solve.inverse
        currentRank = solve.rank
        currentBeta = proposal.clone()
        proposal = addInPlace(proposal.clone(), solve.step)
      }

      logInfo(
        s"CoxPH iteration $iteration: bestLogLik=$bestLogLik, rank=$currentRank, " +
          s"halving=$halving, converged=$converged")
    }

    if (!converged && $(maxIter) > 0) {
      logWarning(s"CoxPH did not converge in ${$(maxIter)} iterations; returning last accepted iterate")
      currentFit = distributedIteration(scaledInstances, currentBeta, method)
      val solve = solveSymmetric(currentFit.information, currentFit.score, $(rankTolerance))
      currentInverse = solve.inverse
      currentRank = solve.rank
      bestLogLik = currentFit.logLik
    }

    if (currentRank < nvar) {
      logWarning(
        s"CoxPH information matrix appears rank-deficient: rank=$currentRank < $nvar. " +
          "Some coefficients may be weakly identified or non-identifiable.")
    }

    val coefficients = new Array[Double](nvar)
    var i = 0
    while (i < nvar) {
      coefficients(i) = currentBeta(i) * scaleArr(i)
      i += 1
    }

    val varianceOriginal = rescaleVariance(currentInverse, scaleArr)
    val baselineHazard = computeBaselineHazard(scaledInstances, currentBeta, method)

    scaledInstances.unpersist()

    val model = new CoxPHRegressionModel(
      uid = uid,
      coefficients = Vectors.dense(coefficients),
      means = Vectors.dense(meansArr),
      featureScales = Vectors.dense(scaleArr),
      varianceMatrix = toSparkMatrix(varianceOriginal),
      baselineHazardTimes = baselineHazard.times,
      cumBaselineHazard = baselineHazard.cumulativeHazard,
      loglikInit = initialLogLik,
      loglikFinal = bestLogLik,
      iterations = iteration,
      converged = converged || $(maxIter) == 0,
      rank = currentRank,
      scoreTest = scoreTest)

    copyValues(model).setParent(this)
  }

  private def computeMeansAndScale(
      instances: RDD[CoxInstance],
      nvar: Int,
      nObs: Long): (Array[Double], Array[Double]) = {

    val sums = instances.treeAggregate(new Array[Double](nvar))(
      seqOp = { (acc, inst) =>
        var j = 0
        while (j < nvar) {
          acc(j) += inst.features(j)
          j += 1
        }
        acc
      },
      combOp = { (left, right) =>
        var j = 0
        while (j < nvar) {
          left(j) += right(j)
          j += 1
        }
        left
      },
      depth = $(aggregationDepth)
    )

    val means = sums.map(_ / nObs.toDouble)
    val bcMeans = instances.sparkContext.broadcast(means)

    val absCenteredSums = instances.treeAggregate(new Array[Double](nvar))(
      seqOp = { (acc, inst) =>
        val localMeans = bcMeans.value
        var j = 0
        while (j < nvar) {
          acc(j) += math.abs(inst.features(j) - localMeans(j))
          j += 1
        }
        acc
      },
      combOp = { (left, right) =>
        var j = 0
        while (j < nvar) {
          left(j) += right(j)
          j += 1
        }
        left
      },
      depth = $(aggregationDepth)
    )

    bcMeans.destroy()

    val scale = new Array[Double](nvar)
    var j = 0
    while (j < nvar) {
      scale(j) =
        if (absCenteredSums(j) > 0.0) nObs.toDouble / absCenteredSums(j)
        else 1.0
      j += 1
    }

    (means, scale)
  }

  private def distributedIteration(
      data: RDD[CoxInstance],
      beta: Array[Double],
      method: Int): IterationResult = {

    val nvar = beta.length
    val triSize = nvar * (nvar + 1) / 2
    val bcBeta = data.sparkContext.broadcast(beta)

    val perTime: Array[(Double, TimeSufficientStats)] = data.mapPartitions { iter =>
      val localBeta = bcBeta.value
      iter.map { inst =>
        val x = inst.features
        var zbeta = 0.0
        var j = 0
        while (j < nvar) {
          zbeta += localBeta(j) * x(j)
          j += 1
        }

        val cappedZbeta = math.max(-MaxSafeExp, math.min(MaxSafeExp, zbeta))
        val hadCapping = cappedZbeta != zbeta
        val risk = math.exp(cappedZbeta)

        val riskX = new Array[Double](nvar)
        j = 0
        while (j < nvar) {
          riskX(j) = risk * x(j)
          j += 1
        }

        val riskXX = new Array[Double](triSize)
        var idx = 0
        var row = 0
        while (row < nvar) {
          var col = 0
          while (col <= row) {
            riskXX(idx) = risk * x(row) * x(col)
            idx += 1
            col += 1
          }
          row += 1
        }

        val deadRisk = if (inst.status == 1) risk else 0.0
        val deadRiskX = if (inst.status == 1) riskX.clone() else new Array[Double](nvar)
        val deadRiskXX = if (inst.status == 1) riskXX.clone() else new Array[Double](triSize)
        val deadX = if (inst.status == 1) x.clone() else new Array[Double](nvar)

        inst.time -> TimeSufficientStats(
          totalRisk = risk,
          totalRiskX = riskX,
          totalRiskXX = riskXX,
          nDead = inst.status,
          deadRisk = deadRisk,
          deadRiskX = deadRiskX,
          deadRiskXX = deadRiskXX,
          deadX = deadX,
          hadNumericalCapping = hadCapping)
      }
    }.reduceByKey(mergeTimeStats).collect().sortBy(_._1)

    bcBeta.destroy()

    var cumS0 = 0.0
    val cumS1 = new Array[Double](nvar)
    val cumS2 = new Array[Double](triSize)

    var logLik = 0.0
    val score = new Array[Double](nvar)
    val info = Array.ofDim[Double](nvar, nvar)
    var hadNumericalIssue = false

    var t = perTime.length - 1
    while (t >= 0) {
      val stats = perTime(t)._2

      cumS0 += stats.totalRisk
      addArrayInPlace(cumS1, stats.totalRiskX)
      addArrayInPlace(cumS2, stats.totalRiskXX)
      hadNumericalIssue = hadNumericalIssue || stats.hadNumericalCapping

      if (stats.nDead > 0) {
        addArrayInPlace(score, stats.deadX)

        var lpDeaths = 0.0
        var j = 0
        while (j < nvar) {
          lpDeaths += beta(j) * stats.deadX(j)
          j += 1
        }
        logLik += lpDeaths

        if (method == 0 || stats.nDead == 1) {
          if (!(cumS0 > 0.0) || !cumS0.isFinite) {
            hadNumericalIssue = true
          } else {
            logLik -= stats.nDead.toDouble * math.log(cumS0)
            j = 0
            while (j < nvar) {
              score(j) -= stats.nDead.toDouble * cumS1(j) / cumS0
              j += 1
            }

            updateInformation(info, cumS0, cumS1, cumS2, stats.nDead.toDouble)
          }
        } else {
          val ndead = stats.nDead.toDouble
          var k = 0
          while (k < stats.nDead) {
            val frac = k.toDouble / ndead
            val adjS0 = cumS0 - frac * stats.deadRisk
            if (!(adjS0 > 0.0) || !adjS0.isFinite) {
              hadNumericalIssue = true
            } else {
              logLik -= math.log(adjS0)
              j = 0
              while (j < nvar) {
                val adjS1 = cumS1(j) - frac * stats.deadRiskX(j)
                score(j) -= adjS1 / adjS0
                j += 1
              }
              updateInformationEfron(
                info,
                adjS0,
                cumS1,
                cumS2,
                stats.deadRiskX,
                stats.deadRiskXX,
                frac)
            }
            k += 1
          }
        }
      }
      t -= 1
    }

    symmetrizeUpperToLower(info)

    IterationResult(
      logLik = if (hadNumericalIssue && !logLik.isFinite) Double.NaN else logLik,
      score = score,
      information = info,
      hadNumericalIssue = hadNumericalIssue)
  }

  private def computeBaselineHazard(
      data: RDD[CoxInstance],
      beta: Array[Double],
      method: Int): BaselineHazard = {

    val bcBeta = data.sparkContext.broadcast(beta)

    val perTime = data.mapPartitions { iter =>
      val localBeta = bcBeta.value
      iter.map { inst =>
        var zbeta = 0.0
        var j = 0
        while (j < localBeta.length) {
          zbeta += localBeta(j) * inst.features(j)
          j += 1
        }
        val risk = math.exp(math.max(-MaxSafeExp, math.min(MaxSafeExp, zbeta)))
        inst.time -> (risk, if (inst.status == 1) risk else 0.0, inst.status)
      }
    }.reduceByKey {
      case ((tr1, dr1, nd1), (tr2, dr2, nd2)) =>
        (tr1 + tr2, dr1 + dr2, nd1 + nd2)
    }.collect().sortBy(_._1)

    bcBeta.destroy()

    val times = ArrayBuffer[Double]()
    val increments = ArrayBuffer[Double]()

    var remainingRisk = 0.0
    var i = 0
    while (i < perTime.length) {
      remainingRisk += perTime(i)._2._1
      i += 1
    }

    var removedRisk = 0.0
    i = 0
    while (i < perTime.length) {
      val (time, (totalRiskAtTime, deadRiskAtTime, nDead)) = perTime(i)
      val riskSetSum = remainingRisk - removedRisk

      if (nDead > 0 && riskSetSum > 0.0 && riskSetSum.isFinite) {
        val increment =
          if (method == 0 || nDead == 1) {
            nDead.toDouble / riskSetSum
          } else {
            var h = 0.0
            var k = 0
            while (k < nDead) {
              h += 1.0 / (riskSetSum - (k.toDouble / nDead.toDouble) * deadRiskAtTime)
              k += 1
            }
            h
          }
        times += time
        increments += increment
      }

      removedRisk += totalRiskAtTime
      i += 1
    }

    val cumulative = increments.scanLeft(0.0)(_ + _).tail.toArray
    BaselineHazard(times.toArray, cumulative)
  }

  private def mergeTimeStats(left: TimeSufficientStats, right: TimeSufficientStats): TimeSufficientStats = {
    addArrayInPlace(left.totalRiskX, right.totalRiskX)
    addArrayInPlace(left.totalRiskXX, right.totalRiskXX)
    addArrayInPlace(left.deadRiskX, right.deadRiskX)
    addArrayInPlace(left.deadRiskXX, right.deadRiskXX)
    addArrayInPlace(left.deadX, right.deadX)

    TimeSufficientStats(
      totalRisk = left.totalRisk + right.totalRisk,
      totalRiskX = left.totalRiskX,
      totalRiskXX = left.totalRiskXX,
      nDead = left.nDead + right.nDead,
      deadRisk = left.deadRisk + right.deadRisk,
      deadRiskX = left.deadRiskX,
      deadRiskXX = left.deadRiskXX,
      deadX = left.deadX,
      hadNumericalCapping = left.hadNumericalCapping || right.hadNumericalCapping)
  }

  private def updateInformation(
      info: Array[Array[Double]],
      s0: Double,
      s1: Array[Double],
      s2: Array[Double],
      weight: Double): Unit = {

    val invS0 = 1.0 / s0
    var idx = 0
    var i = 0
    while (i < s1.length) {
      val meanI = s1(i) * invS0
      var j = 0
      while (j <= i) {
        val meanJ = s1(j) * invS0
        info(j)(i) += weight * (s2(idx) * invS0 - meanI * meanJ)
        idx += 1
        j += 1
      }
      i += 1
    }
  }

  private def updateInformationEfron(
      info: Array[Array[Double]],
      adjS0: Double,
      cumS1: Array[Double],
      cumS2: Array[Double],
      deadS1: Array[Double],
      deadS2: Array[Double],
      frac: Double): Unit = {

    val invAdjS0 = 1.0 / adjS0
    var idx = 0
    var i = 0
    while (i < cumS1.length) {
      val adjS1i = cumS1(i) - frac * deadS1(i)
      val meanI = adjS1i * invAdjS0
      var j = 0
      while (j <= i) {
        val adjS1j = cumS1(j) - frac * deadS1(j)
        val meanJ = adjS1j * invAdjS0
        val adjS2ij = cumS2(idx) - frac * deadS2(idx)
        info(j)(i) += adjS2ij * invAdjS0 - meanI * meanJ
        idx += 1
        j += 1
      }
      i += 1
    }
  }

  private def solveSymmetric(
      information: Array[Array[Double]],
      score: Array[Double],
      relTol: Double): LinearSolveResult = {

    val n = score.length
    val dense = new BDM[Double](n, n)
    var i = 0
    while (i < n) {
      var j = 0
      while (j < n) {
        dense(i, j) = information(i)(j)
        j += 1
      }
      i += 1
    }

    val eigen = eigSym(dense)
    val eigenValues = eigen.eigenvalues
    val eigenVectors = eigen.eigenvectors

    var maxAbsEigen = 0.0
    i = 0
    while (i < n) {
      maxAbsEigen = math.max(maxAbsEigen, math.abs(eigenValues(i)))
      i += 1
    }
    val absTol = math.max(relTol * maxAbsEigen, relTol)

    val invDiag = Array.fill[Double](n)(0.0)
    var rank = 0
    i = 0
    while (i < n) {
      if (eigenValues(i) > absTol) {
        invDiag(i) = 1.0 / eigenValues(i)
        rank += 1
      }
      i += 1
    }

    val scoreVec = new BDV[Double](score)
    val rotatedScore = eigenVectors.t * scoreVec
    i = 0
    while (i < n) {
      rotatedScore(i) *= invDiag(i)
      i += 1
    }

    val stepVec = eigenVectors * rotatedScore
    val step = stepVec.toArray

    val inverse = Array.ofDim[Double](n, n)
    var col = 0
    while (col < n) {
      var row = 0
      while (row < n) {
        var acc = 0.0
        var k = 0
        while (k < n) {
          acc += eigenVectors(row, k) * invDiag(k) * eigenVectors(col, k)
          k += 1
        }
        inverse(row)(col) = acc
        row += 1
      }
      col += 1
    }

    LinearSolveResult(step = step, inverse = inverse, rank = rank)
  }

  private def rescaleVariance(
      varianceScaled: Array[Array[Double]],
      scale: Array[Double]): Array[Array[Double]] = {
    val n = scale.length
    val varianceOriginal = Array.ofDim[Double](n, n)
    var i = 0
    while (i < n) {
      var j = 0
      while (j < n) {
        varianceOriginal(i)(j) = varianceScaled(i)(j) * scale(i) * scale(j)
        j += 1
      }
      i += 1
    }
    varianceOriginal
  }

  private def quadraticForm(x: Array[Double], y: Array[Double]): Double = {
    var out = 0.0
    var i = 0
    while (i < x.length) {
      out += x(i) * y(i)
      i += 1
    }
    out
  }

  private def toSparkMatrix(values: Array[Array[Double]]): Matrix = {
    val nRows = values.length
    val nCols = if (nRows == 0) 0 else values(0).length
    val columnMajor = Array.ofDim[Double](nRows * nCols)
    var i = 0
    while (i < nRows) {
      var j = 0
      while (j < nCols) {
        columnMajor(j * nRows + i) = values(i)(j)
        j += 1
      }
      i += 1
    }
    Matrices.dense(nRows, nCols, columnMajor)
  }

  private def relativeLogLikChange(previous: Double, current: Double): Double = {
    if (current == 0.0) math.abs(previous - current)
    else math.abs(1.0 - previous / current)
  }

  private def addArrayInPlace(left: Array[Double], right: Array[Double]): Unit = {
    var i = 0
    while (i < left.length) {
      left(i) += right(i)
      i += 1
    }
  }

  private def addInPlace(left: Array[Double], right: Array[Double]): Array[Double] = {
    var i = 0
    while (i < left.length) {
      left(i) += right(i)
      i += 1
    }
    left
  }

  private def allFinite(values: Array[Double]): Boolean = values.forall(_.isFinite)

  private def allFinite(matrix: Array[Array[Double]]): Boolean = matrix.forall(row => row.forall(_.isFinite))

  private def symmetrizeUpperToLower(matrix: Array[Array[Double]]): Unit = {
    var i = 0
    while (i < matrix.length) {
      var j = 0
      while (j < i) {
        matrix(i)(j) = matrix(j)(i)
        j += 1
      }
      i += 1
    }
  }

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema, fitting = true)

  override def copy(extra: ParamMap): CoxPHRegression = defaultCopy(extra)
}

class CoxPHRegressionModel private[ml](
    override val uid: String,
    val coefficients: Vector,
    val means: Vector,
    val featureScales: Vector,
    val varianceMatrix: Matrix,
    val baselineHazardTimes: Array[Double],
    val cumBaselineHazard: Array[Double],
    val loglikInit: Double,
    val loglikFinal: Double,
    val iterations: Int,
    val converged: Boolean,
    val rank: Int,
    val scoreTest: Double)
  extends RegressionModel[Vector, CoxPHRegressionModel]
    with CoxPHRegressionParams with MLWritable {

  override def numFeatures: Int = coefficients.size

  def standardErrors: Vector = {
    val out = new Array[Double](numFeatures)
    var i = 0
    while (i < numFeatures) {
      val v = varianceMatrix(i, i)
      out(i) = if (v >= 0.0) math.sqrt(v) else Double.NaN
      i += 1
    }
    Vectors.dense(out)
  }

  def zStatistics: Vector = {
    val coef = coefficients.toArray
    val se = standardErrors.toArray
    val out = new Array[Double](numFeatures)
    var i = 0
    while (i < numFeatures) {
      out(i) =
        if (se(i) > 0.0 && se(i).isFinite) coef(i) / se(i)
        else Double.NaN
      i += 1
    }
    Vectors.dense(out)
  }

  def predictLinearPredictor(features: Vector): Double = {
    val x = features.toArray
    val m = means.toArray
    val b = coefficients.toArray
    var lp = 0.0
    var i = 0
    while (i < numFeatures) {
      lp += b(i) * (x(i) - m(i))
      i += 1
    }
    lp
  }

  def predictRelativeRisk(features: Vector): Double = math.exp(predictLinearPredictor(features))

  def predictRisk(features: Vector): Double = predictRelativeRisk(features)

  def predictSurvival(features: Vector, time: Double): Double = {
    val h0 = cumulativeBaselineHazardAt(time)
    math.exp(-h0 * predictRelativeRisk(features))
  }

  def predictFailureProbability(features: Vector, time: Double): Double =
    1.0 - predictSurvival(features, time)

  def cumulativeBaselineHazardAt(time: Double): Double = {
    if (baselineHazardTimes.isEmpty || time < baselineHazardTimes(0)) {
      0.0
    } else {
      val idx = java.util.Arrays.binarySearch(baselineHazardTimes, time)
      if (idx >= 0) {
        cumBaselineHazard(idx)
      } else {
        val insertionPoint = -(idx + 1)
        if (insertionPoint <= 0) 0.0
        else if (insertionPoint >= baselineHazardTimes.length) cumBaselineHazard.last
        else cumBaselineHazard(insertionPoint - 1)
      }
    }
  }

  override def predict(features: Vector): Double = predictRelativeRisk(features)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf { features: Vector => predictRelativeRisk(features) }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def transformSchema(schema: StructType): StructType =
    validateAndTransformSchema(schema, fitting = false)

  override def copy(extra: ParamMap): CoxPHRegressionModel = {
    val copied = new CoxPHRegressionModel(
      uid = uid,
      coefficients = coefficients,
      means = means,
      featureScales = featureScales,
      varianceMatrix = varianceMatrix,
      baselineHazardTimes = baselineHazardTimes.clone(),
      cumBaselineHazard = cumBaselineHazard.clone(),
      loglikInit = loglikInit,
      loglikFinal = loglikFinal,
      iterations = iterations,
      converged = converged,
      rank = rank,
      scoreTest = scoreTest)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter =
    throw new UnsupportedOperationException(
      "CoxPHRegressionModel persistence is not implemented yet. Save coefficients, means, " +
        "variance matrix and baseline hazard explicitly.")

  override def toString: String =
    s"CoxPHRegressionModel(uid=$uid, numFeatures=$numFeatures, converged=$converged, " +
      s"iterations=$iterations, rank=$rank, loglik=($loglikInit, $loglikFinal))"
}
