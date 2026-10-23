/*
 * Copyright (C) 2025 AxionOS
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package co.aospa.sense.ml

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import co.aospa.sense.util.GpuDelegateHelper
import co.aospa.sense.util.Loggable
import co.aospa.sense.util.logD
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.DataType
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.HashMap
import kotlinx.coroutines.*
import kotlin.math.abs

class AntiSpoofModel(context: Context) : Loggable {

    private val interpreter: Interpreter
    private var gpuDelegate: Delegate? = null
    private var nnApiDelegate: NnApiDelegate? = null
    
    private val imageTensorProcessor = ImageProcessor.Builder()
        .add(ResizeOp(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(0f, 255f))
        .build()

    init {
        val options = Interpreter.Options()
        
        gpuDelegate = GpuDelegateHelper.createGpuDelegate()
        if (gpuDelegate != null) {
            options.addDelegate(gpuDelegate)
            logD("AntiSpoofModel: Applied GPU Delegate")
        } else {
            try {
                nnApiDelegate = NnApiDelegate()
                options.addDelegate(nnApiDelegate)
                logD("AntiSpoofModel: Applied NNAPI Delegate")
            } catch (e: Exception) {
                logD("AntiSpoofModel: NNAPI Delegate unavailable: ${e.message}")
                options.numThreads = 4
                logD("AntiSpoofModel: Fallback to CPU with 4 threads")
            }
        }
        
        interpreter = Interpreter(FileUtil.loadMappedFile(context, MODEL_FILE), options)
    }

    data class Result(val isSpoof: Boolean, val score: Float)

    suspend fun detect(bitmap: Bitmap): Result = withContext(Dispatchers.IO) {
        val bitmapScale = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true)

        val laplacianScore = laplacian(bitmapScale)
        logD("detect: laplacian=$laplacianScore")

        if (laplacianScore < LAPLACIAN_THRESHOLD) {
            logD("detect: Image too blurry, rejecting as spoof.")
            return@withContext Result(isSpoof = true, score = 0f)
        }

        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)
        val processedImage = imageTensorProcessor.process(tensorImage)
        
        val inputBuffer = processedImage.buffer

        val clss_pred = Array(1) { FloatArray(8) }
        val leaf_node_mask = Array(1) { FloatArray(8) }
        
        val outputs = HashMap<Int, Any>()
        outputs[interpreter.getOutputIndex("Identity")] = clss_pred
        outputs[interpreter.getOutputIndex("Identity_1")] = leaf_node_mask
        
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        logD("detect: clss_pred: [" + clss_pred[0][0] + ", " + clss_pred[0][1] + ", "
                + clss_pred[0][2] + ", " + clss_pred[0][3] + ", " + clss_pred[0][4] + ", "
                + clss_pred[0][5] + ", " + clss_pred[0][6] + ", " + clss_pred[0][7] + "]")
        logD("detect: leaf_node_mask: [" + leaf_node_mask[0][0] + ", " + leaf_node_mask[0][1] + ", "
                + leaf_node_mask[0][2] + ", " + leaf_node_mask[0][3] + ", " + leaf_node_mask[0][4] + ", "
                + leaf_node_mask[0][5] + ", " + leaf_node_mask[0][6] + ", " + leaf_node_mask[0][7] + "]")

        val score = leaf_score1(clss_pred, leaf_node_mask)

        val isSpoof = score > THRESHOLD

        logD("detect: isSpoof=$isSpoof, score=$score")

        Result(isSpoof, score)
    }

    private fun leaf_score1(clss_pred: Array<FloatArray>, leaf_node_mask: Array<FloatArray>): Float {
        var score = 0f
        for (i in 0 until 8) {
            score += abs(clss_pred[0][i]) * leaf_node_mask[0][i]
        }
        return score
    }

    fun laplacian(bitmap: Bitmap): Int {
        val width = bitmap.width
        val height = bitmap.height
        if (width < 3 || height < 3) return 0

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val grey = IntArray(pixels.size)
        for (i in pixels.indices) {
            val color = pixels[i]
            val red = (color shr 16) and 0xFF
            val green = (color shr 8) and 0xFF
            val blue = color and 0xFF
            grey[i] = (red * 0.3f + green * 0.59f + blue * 0.11f).toInt()
        }

        var score = 0
        for (x in 0..height - 3) {
            val top = x * width
            val middle = (x + 1) * width
            val bottom = (x + 2) * width
            for (y in 0..width - 3) {
                val result = grey[top + y + 1] +
                        grey[middle + y] +
                        grey[middle + y + 2] +
                        grey[bottom + y + 1] -
                        4 * grey[middle + y + 1]
                if (result > LAPLACE_THRESHOLD) {
                    score++
                }
            }
        }
        return score
    }

    fun close() {
        interpreter.close()
        (gpuDelegate as? AutoCloseable)?.close()
        nnApiDelegate?.close()
    }

    companion object {
        private const val MODEL_FILE = "face_anti_spoofing.tflite"

        const val INPUT_IMAGE_SIZE = 256
        const val THRESHOLD = 0.1f

        const val ROUTE_INDEX = 6

        const val LAPLACE_THRESHOLD = 50
        const val LAPLACIAN_THRESHOLD = 500
    }
}
