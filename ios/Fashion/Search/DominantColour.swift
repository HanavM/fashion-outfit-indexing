import SwiftUI
import UIKit
import CoreImage

/// Pulls a single RGB triple out of a photograph, for "pick a colour from a
/// photo".
///
/// Deliberately the *average* rather than a clustered dominant colour. The
/// backend takes a single `colour_rgb` and matches against its own colour
/// model; a k-means palette here would be a second, different colour opinion
/// on the client, and when the two disagreed the user would have no way to
/// tell which one produced the results. The UI therefore shows the swatch it
/// extracted, so what was picked is visible before it is used.
enum DominantColour {
    private static let context = CIContext(options: [.workingColorSpace: NSNull()])

    static func average(of image: UIImage) -> [Int]? {
        guard let cgImage = image.cgImage else { return nil }
        let input = CIImage(cgImage: cgImage)
        let extent = CIVector(x: input.extent.origin.x, y: input.extent.origin.y,
                              z: input.extent.size.width, w: input.extent.size.height)

        guard let filter = CIFilter(name: "CIAreaAverage",
                                    parameters: [kCIInputImageKey: input,
                                                 kCIInputExtentKey: extent]),
              let output = filter.outputImage else { return nil }

        var bitmap = [UInt8](repeating: 0, count: 4)
        context.render(output,
                       toBitmap: &bitmap,
                       rowBytes: 4,
                       bounds: CGRect(x: 0, y: 0, width: 1, height: 1),
                       format: .RGBA8,
                       colorSpace: CGColorSpaceCreateDeviceRGB())

        return [Int(bitmap[0]), Int(bitmap[1]), Int(bitmap[2])]
    }

    /// A `Color` for the swatch. This is the one place the app renders a
    /// non-semantic colour, and it is unavoidable: the swatch's entire job is
    /// to show the user the literal colour that will be sent. It is always
    /// accompanied by text, never used to convey state, and never becomes a
    /// second accent.
    static func swatch(_ rgb: [Int]) -> Color {
        guard rgb.count >= 3 else { return .clear }
        return Color(.sRGB,
                     red: Double(rgb[0]) / 255,
                     green: Double(rgb[1]) / 255,
                     blue: Double(rgb[2]) / 255)
    }

    /// A words-only description, so the swatch is never the sole carrier of
    /// meaning for VoiceOver or for anyone who can't distinguish it.
    static func description(_ rgb: [Int]) -> String {
        guard rgb.count >= 3 else { return "no colour" }
        return "red \(rgb[0]), green \(rgb[1]), blue \(rgb[2])"
    }
}
