import SwiftUI
import UIKit

/// One component of a multi-part query.
///
/// A query is any number of photos and any number of phrases, and each part
/// is matched against a *different* garment in the result photo. That is the
/// whole point of the product — "this jacket, with baggy jeans" is two parts
/// claiming two garments, not one blended query — so parts are modelled
/// individually rather than as a single string plus a single image.
enum QueryPart: Identifiable, Equatable {
    case image(id: UUID, image: UIImage, caption: String?)
    case text(id: UUID, value: String)

    var id: UUID {
        switch self {
        case .image(let id, _, _): return id
        case .text(let id, _): return id
        }
    }

    var displayText: String {
        switch self {
        case .image(_, _, let caption): return caption ?? "Photo"
        case .text(_, let value): return value
        }
    }

    static func image(_ image: UIImage, caption: String? = nil) -> QueryPart {
        .image(id: UUID(), image: image, caption: caption)
    }

    static func text(_ value: String) -> QueryPart {
        .text(id: UUID(), value: value)
    }
}

/// A hand-off buffer for query parts staged from another screen.
///
/// The closet's "see how this looks with…" needs to put a garment photo into
/// Search while switching tabs. Passing it through the router would make the
/// router know about images; a small dedicated buffer keeps that separation.
@MainActor
@Observable
final class SearchComposition {
    static let shared = SearchComposition()

    private(set) var staged: [QueryPart] = []

    func stage(image: UIImage, caption: String?) {
        staged = [.image(image, caption: caption)]
    }

    func consume() -> [QueryPart] {
        defer { staged = [] }
        return staged
    }

    var hasStaged: Bool { !staged.isEmpty }
}

/// Everything the filter row can constrain.
struct SearchFilters: Equatable {
    var colourName: String?
    /// Set by "pick from a photo". Takes precedence over `colourName`.
    var colourRGB: [Int]?
    var category: String?
    var usOnly: Bool = false
    var mensOnly: Bool = false

    /// Nil means the control is off — no skin constraint is sent at all.
    /// Defaulting it to 0.5 would silently apply a filter nobody asked for.
    var skinTone: Double?
    /// A reference photo, which the backend reads directly. Preferred over
    /// the slider because it asks the user for an example rather than for a
    /// position on a scale.
    var skinReferenceImage: UIImage?

    var isActive: Bool {
        colourName != nil || colourRGB != nil || category != nil
            || usOnly || mensOnly || skinTone != nil || skinReferenceImage != nil
    }

    /// A short spoken/written summary, used for the filter button's label and
    /// its VoiceOver description.
    var summary: String {
        var parts: [String] = []
        if let colourName { parts.append(colourName) }
        if colourRGB != nil { parts.append("colour from a photo") }
        if let category { parts.append(category) }
        if usOnly { parts.append("US only") }
        if mensOnly { parts.append("men’s only") }
        if skinReferenceImage != nil { parts.append("skin from a photo") }
        else if skinTone != nil { parts.append("skin similarity") }
        return parts.isEmpty ? "Filters" : parts.joined(separator: ", ")
    }
}
