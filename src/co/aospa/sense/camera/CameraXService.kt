/*
 * Copyright (C) 2025 AxionOS
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package co.aospa.sense.camera

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.os.Handler
import android.os.Looper
import android.util.Size
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import co.aospa.sense.util.Loggable
import co.aospa.sense.util.logD
import java.nio.ByteBuffer
import java.util.concurrent.Executors
import android.os.SystemProperties
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.ExperimentalCamera2Interop

class CameraXService(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner
) : Loggable {

    interface FrameCallback {
        fun onFrame(bitmap: Bitmap)
        fun onError(error: Exception)
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private var cameraProvider: ProcessCameraProvider? = null
    private var imageAnalysis: ImageAnalysis? = null
    private var preview: Preview? = null
    private var callback: FrameCallback? = null
    private var isProcessing = false
    private val analyzerExecutor = Executors.newSingleThreadExecutor()

    private var frameBitmap: Bitmap? = null
    private var transformedBitmap: Bitmap? = null
    private var transformedCanvas: Canvas? = null
    private var packedBuffer: ByteBuffer? = null
    private var rowBuffer: ByteArray? = null
    private val transformMatrix = Matrix()
    private val transformPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private var transformRotation = Int.MIN_VALUE
    private var surfaceProvider: Preview.SurfaceProvider? = null
    private var isBound = false

    private val targetSize: Size by lazy { Size(640, 480) }

    fun setSurfaceProvider(provider: Preview.SurfaceProvider?) {
        val hadProvider = surfaceProvider != null
        surfaceProvider = provider
        
        if (provider != null && isBound && preview != null) {
            preview?.surfaceProvider = provider
        } else if (provider != null && cameraProvider != null && !hadProvider) {
            mainHandler.post { rebindWithPreview() }
        }
    }

    fun start(frameCallback: FrameCallback) {
        callback = frameCallback
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
        
        cameraProviderFuture.addListener({
            try {
                cameraProvider = cameraProviderFuture.get()
                bindCamera()
            } catch (e: Exception) {
                logD("Failed to get camera provider: ${e.message}")
                callback?.onError(e)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    @ExperimentalCamera2Interop
    private fun getCameraSelector(): CameraSelector {
        val cameraId = SystemProperties.get("ro.face.sense_service.camera_id", "")
        if (cameraId.isNotEmpty()) {
            logD("Requested camera ID from prop: $cameraId")
            return CameraSelector.Builder()
                .addCameraFilter { cameraInfos ->
                    val allIds = cameraInfos.map { Camera2CameraInfo.from(it).cameraId }
                    logD("Available camera IDs: $allIds")
                    
                    val filtered = cameraInfos.filter {
                        try {
                            val infoId = Camera2CameraInfo.from(it).cameraId
                            infoId == cameraId || infoId.endsWith(":$cameraId")
                        } catch (e: Exception) {
                            false
                        }
                    }
                    
                    if (filtered.isEmpty()) {
                        logD("Camera ID $cameraId not found in available list, searching for front camera...")
                        val frontCameras = cameraInfos.filter {
                            it.lensFacing == CameraSelector.LENS_FACING_FRONT
                        }
                        if (frontCameras.isEmpty()) {
                            logD("No front camera found, using first available camera")
                            cameraInfos.take(1)
                        } else {
                            frontCameras
                        }
                    } else {
                        logD("Selected camera ID(s): ${filtered.map { Camera2CameraInfo.from(it).cameraId }}")
                        filtered
                    }
                }
                .build()
        }

        logD("No camera ID prop, using default front camera")
        return CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
            .build()
    }

    @ExperimentalCamera2Interop
    private fun bindCamera() {
        val provider = cameraProvider ?: return
        val cameraSelector = getCameraSelector()

        imageAnalysis = createImageAnalysis()
        imageAnalysis?.setAnalyzer(analyzerExecutor, analyzer)

        try {
            provider.unbindAll()
            if (surfaceProvider != null) {
                preview = Preview.Builder()
                    .setTargetResolution(targetSize)
                    .build()
                preview?.surfaceProvider = surfaceProvider
                provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
            } else {
                provider.bindToLifecycle(lifecycleOwner, cameraSelector, imageAnalysis)
            }

            isBound = true
            logD("Camera bound successfully, preview=${surfaceProvider != null}")
        } catch (e: Exception) {
            logD("Failed to bind camera: ${e.message}")
            callback?.onError(e)
        }
    }

    @ExperimentalCamera2Interop
    private fun rebindWithPreview() {
        val provider = cameraProvider ?: return
        val cameraSelector = getCameraSelector()

        try {
            provider.unbindAll()
            preview = Preview.Builder()
                .setTargetResolution(targetSize)
                .build()
            preview?.surfaceProvider = surfaceProvider

            imageAnalysis = createImageAnalysis()
            imageAnalysis?.setAnalyzer(analyzerExecutor, analyzer)

            provider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
            isBound = true
            logD("Camera rebound with preview")
        } catch (e: Exception) {
            logD("Failed to rebind camera: ${e.message}")
        }
    }

    private fun createImageAnalysis(): ImageAnalysis {
        return ImageAnalysis.Builder()
            .setTargetResolution(targetSize)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
    }

    @ExperimentalGetImage
    private val analyzer = ImageAnalysis.Analyzer { image ->
        if (isProcessing) {
            image.close()
            return@Analyzer
        }
        isProcessing = true

        try {
             val plane = image.planes[0]
             val rowStride = plane.rowStride
             val pixelStride = plane.pixelStride
             val width = image.width
             val height = image.height
             val buffer = plane.buffer
             buffer.rewind()

             var src = frameBitmap
             if (src == null || src.width != width || src.height != height) {
                 src = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                 frameBitmap = src
                 transformRotation = Int.MIN_VALUE
             }

             val rowBytes = width * pixelStride
             if (rowStride == rowBytes) {
                 src.copyPixelsFromBuffer(buffer)
             } else {
                 val packedSize = rowBytes * height
                 var packed = packedBuffer
                 if (packed == null || packed.capacity() < packedSize) {
                     packed = ByteBuffer.allocate(packedSize)
                     packedBuffer = packed
                 }
                 packed.clear()

                 var row = rowBuffer
                 if (row == null || row.size < rowBytes) {
                     row = ByteArray(rowBytes)
                     rowBuffer = row
                 }

                 for (y in 0 until height) {
                     buffer.position(y * rowStride)
                     buffer.get(row, 0, rowBytes)
                     packed.put(row, 0, rowBytes)
                 }
                 packed.rewind()
                 src.copyPixelsFromBuffer(packed)
             }

            val rotation = image.imageInfo.rotationDegrees
            val rotatedWidth = if (rotation == 90 || rotation == 270) height else width
            val rotatedHeight = if (rotation == 90 || rotation == 270) width else height

            if (transformRotation != rotation) {
                transformMatrix.reset()
                transformMatrix.postRotate(rotation.toFloat(), width / 2f, height / 2f)
                transformMatrix.postScale(-1f, 1f, width / 2f, height / 2f)

                val bounds = RectF(0f, 0f, width.toFloat(), height.toFloat())
                transformMatrix.mapRect(bounds)
                transformMatrix.postTranslate(-bounds.left, -bounds.top)
                transformRotation = rotation
            }

            var dst = transformedBitmap
            if (dst == null || dst.width != rotatedWidth || dst.height != rotatedHeight) {
                dst = Bitmap.createBitmap(rotatedWidth, rotatedHeight, Bitmap.Config.ARGB_8888)
                transformedBitmap = dst
                transformedCanvas = Canvas(dst)
            }

            transformedCanvas!!.drawBitmap(src, transformMatrix, transformPaint)
            callback?.onFrame(dst!!)
        } catch (e: Exception) {
            logD("Frame processing error: ${e.message}")
        } finally {
            isProcessing = false
            image.close()
        }
    }

    fun stop() {
        logD("Stopping camera")
        mainHandler.post {
            isBound = false
            cameraProvider = null
            imageAnalysis = null
            preview = null
            callback = null
            frameBitmap = null
            transformedBitmap = null
            transformedCanvas = null
            packedBuffer = null
            rowBuffer = null
            transformRotation = Int.MIN_VALUE
            surfaceProvider = null
            try {
                if (!analyzerExecutor.isShutdown) {
                    analyzerExecutor.shutdown()
                }
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    fun resetProcessing() {
        isProcessing = false
    }
}
