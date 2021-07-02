//
//  tflitePredictor.swift
//  MNIST
//
//  Created by sxb on 2021/6/30.
//

import Foundation
import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

typealias FileInfo = (name: String, extension: String)

class tflitePredictor: NSObject {
    // MARK: - Internal Properties
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount: Int
    let threadCountLimit = 10

    init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
        let modelFilename = modelFileInfo.name

        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }

        // Specify the options for the `Interpreter`.
        var coremlOptions = CoreMLDelegate.Options()
        coremlOptions.enabledDevices = .all
        coremlOptions.coreMLVersion = 2
        let coreMLDelegate = CoreMLDelegate(options: coremlOptions)!
      
        var options = Interpreter.Options()
        options.threadCount = threadCount
        self.threadCount = threadCount
        do {
        // Create the `Interpreter`.
            do{
                interpreter = try Interpreter(modelPath: modelPath, options: options, delegates: [coreMLDelegate])
            } catch let error{
                interpreter = try Interpreter(modelPath: modelPath, options: options)
                print(error.localizedDescription)
                print("Interpreter with Metal GPU Failed. Falling back to CPU")
            }
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
      
        super.init()
    }
    
    
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Int? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        
        let preInterval: TimeInterval
        let inferenceInterval: TimeInterval
        let postInterval: TimeInterval
      
        let featMap01: Tensor

        do {
            try interpreter.resizeInput(at: 0, to: Tensor.Shape([1,imageHeight,imageWidth,1]))
            try interpreter.allocateTensors()

            let inputTensor = try interpreter.input(at: 0)
            let startPre = Date()
            guard let rgbData = yDataFromBuffer(
                pixelBuffer,
                byteCount:  imageWidth * imageHeight ,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
        
            let float0 = rgbData.withUnsafeBytes( {
                (pointer: UnsafePointer<Float32>) -> [Float32] in
            let buffer = UnsafeBufferPointer(start: pointer,
                                                 count: rgbData.count/4)
                return Array<Float32>(buffer)
            })

            preInterval = Date().timeIntervalSince(startPre) * 1000
       
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
        
            // Run inference by invoking the `Interpreter`.
            let startDate = Date()
            try interpreter.invoke()
            inferenceInterval = Date().timeIntervalSince(startDate) * 1000
        
            let startPP = Date()

            postInterval = Date().timeIntervalSince(startPP) * 1000
     
            print(String(format: "%02d (FPS), Run Time : %1fms, Pre-Processing Time: %.1fms, Inference Time: %.1fms, Post Process Time: ?.1fms -> %.1fms", Int(1000/(preInterval+inferenceInterval)), preInterval+inferenceInterval, preInterval, inferenceInterval, postInterval))
        
            featMap01 = try interpreter.output(at: 0)

            let floatA = featMap01.data.withUnsafeBytes( {
                (pointer: UnsafePointer<Float32>) -> [Float32] in
                let buffer = UnsafeBufferPointer(start: pointer,
                                                 count: featMap01.data.count / 4)
                return Array<Float32>(buffer)
            })

        
            var tmp = 0.0 as Float32
            var tmpinx = 0
            for index in 0..<10 {
                let item = floatA[index]
                if item > tmp {
                    tmp = item
                    tmpinx = index
                }
            }
            return tmpinx
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return 0
        }
    }
    
    private func yDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
      
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
      
        var byteData = Data()
        let source = Data(bytes: sourceData, count: sourceBytesPerRow * height)

        for index in 0..<height {
            let rowData = source.subdata(in: Range(uncheckedBounds: (index * sourceBytesPerRow, index*sourceBytesPerRow + width)))
            byteData.append(rowData)
        }
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)

        if isModelQuantized {
            return byteData
        }
        let floats = byteData.map{ Float32($0) }
        return Data(copyingBufferOf: floats)
    }
}

// MARK: - Data
extension Data {
 
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }

    func toArray<T>(type: T.Type) -> [T] where T: AdditiveArithmetic {
        var array = [T](repeating: T.zero, count: self.count / MemoryLayout<T>.stride)
        _ = array.withUnsafeMutableBytes { self.copyBytes(to: $0) }
        return array
    }
}
