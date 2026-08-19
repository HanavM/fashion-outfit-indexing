import UIKit

enum ImageEncoding {
    /// Longest edge sent to the backend.
    ///
    /// The encoder works at 384px. A 12-megapixel phone photo is ~4MB of
    /// base64 for no gain in accuracy and a real cost in upload time on a
    /// cold container, so images are downscaled before encoding. 1024 leaves
    /// comfortable headroom above the model's input size.
    static let maximumEdge: CGFloat = 1024
    static let jpegQuality: CGFloat = 0.85

    /// Downscale and re-encode. Also normalises orientation: `UIImage`
    /// carries EXIF rotation that raw JPEG bytes preserve but many decoders
    /// ignore, which is how a correctly-framed photo arrives sideways at a
    /// model.
    static func prepared(_ image: UIImage) -> Data? {
        let size = image.size
        guard size.width > 0, size.height > 0 else { return nil }

        let scale = min(1, maximumEdge / max(size.width, size.height))
        let target = CGSize(width: (size.width * scale).rounded(),
                            height: (size.height * scale).rounded())

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: target, format: format)
        let redrawn = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: target))
        }
        return redrawn.jpegData(compressionQuality: jpegQuality)
    }

    static func base64(_ image: UIImage) -> String? {
        prepared(image)?.base64EncodedString()
    }

    static func base64(_ data: Data) -> String {
        data.base64EncodedString()
    }
}
