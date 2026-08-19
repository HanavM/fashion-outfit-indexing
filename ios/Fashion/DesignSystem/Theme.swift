import SwiftUI

/// The whole visual vocabulary of the app, in one file.
///
/// The rules this encodes, and which every view is expected to obey:
///
/// - **Content leads, chrome recedes.** The content here is photographs of
///   people. Views give them the full frame and keep everything else quiet.
/// - **Light is the default.** There is no dark hero surface anywhere. Dark
///   mode is supported because semantic colours provide it, not designed for.
/// - **One accent.** `Color.accentColor`, defined once in the asset catalog
///   with light / dark / increased-contrast variants. No view introduces a
///   second accent. Colour is never the only carrier of meaning.
/// - **No hardcoded colours in code.** Everything below resolves to a
///   semantic system colour or to the accent, so Dark Mode, Increase
///   Contrast and Smart Invert all keep working for free.
/// - **Hierarchy comes from type and space**, not from boxes, borders or
///   glow. There is deliberately no "card" style with a stroke and a shadow.
enum Theme {

    /// A 4pt-derived spacing scale. Named by intent, not by number, so a
    /// view reads as its layout rather than as arithmetic.
    enum Space {
        /// Between a label and the thing it labels.
        static let hairline: CGFloat = 2
        /// Within a single unit of content.
        static let tight: CGFloat = 6
        /// The default gap between related elements.
        static let snug: CGFloat = 12
        /// Standard screen inset, matching the system's own margins.
        static let margin: CGFloat = 16
        /// Between distinct groups.
        static let section: CGFloat = 28
    }

    enum Radius {
        /// Photographs. Small enough to read as a print, not a bubble.
        static let photo: CGFloat = 12
        /// Controls that need a shape of their own.
        static let control: CGFloat = 10
    }

    /// Apple's minimum comfortable hit target. Any control smaller than this
    /// visually must still claim this much space.
    static let minimumTouchTarget: CGFloat = 44

    /// Motion.
    ///
    /// This is a utility app. Motion exists to explain what moved where and
    /// nothing else — no cinematic reveals, no staged cascades. Springs are
    /// short (<= 300ms) and interruptible, and every one of them is routed
    /// through `Motion.standard(reduceMotion:)` so that Reduce Motion
    /// collapses it to no animation at all rather than to a faster one.
    enum Motion {
        static func standard(reduceMotion: Bool) -> Animation? {
            reduceMotion ? nil : .spring(response: 0.28, dampingFraction: 0.9)
        }

        /// For changes that swap content wholesale (a results grid arriving).
        /// Slightly softer, still under the budget.
        static func content(reduceMotion: Bool) -> Animation? {
            reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.95)
        }
    }
}

extension View {
    /// Guarantees a 44pt touch target without forcing the *visual* element
    /// to be 44pt, which is the usual reason this rule gets broken.
    func minimumHitTarget() -> some View {
        contentShape(Rectangle())
            .frame(minWidth: Theme.minimumTouchTarget,
                   minHeight: Theme.minimumTouchTarget)
    }
}

/// A short piece of text that qualifies the confidence of what is next to it.
///
/// This exists because the product has real, measured limits that must not be
/// designed away: the catalog is 20 US men's brands and will confidently name
/// the closest thing it knows to anything else; the garment labels on outfit
/// photos are unvalidated model output. Those facts belong beside the claim,
/// every time, so this is a component rather than an ad-hoc `Text`.
///
/// Rendered as secondary footnote text, not as a warning badge — it is
/// context, not an error, and colouring it would spend a second accent.
struct Qualifier: View {
    private let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        Text(text)
            .font(.footnote)
            .foregroundStyle(.secondary)
            // Never truncate a caveat. It wraps as far as it needs to.
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityLabel(text)
    }
}

/// A match score, shown as a number rather than as a bar or a badge.
///
/// Deliberately plain. A five-star widget or a green/amber/red pill would
/// imply a calibration this model does not have — the wrong answer has
/// outscored the right one 0.922 to 0.854 in this system's own evaluation.
/// So the number is shown, the scale is named, and no visual encoding
/// promises more than that.
struct ScoreLabel: View {
    private let score: Double
    private let scale: Scale

    enum Scale {
        /// Cosine-style similarity in roughly 0...1.
        case similarity
        /// The catalog text route returns an unbounded integer keyword
        /// score. Rendering that as a percentage would be a lie.
        case keywordMatch
    }

    init(_ score: Double, scale: Scale = .similarity) {
        self.score = score
        self.scale = scale
    }

    private var formatted: String {
        switch scale {
        case .similarity:
            return String(format: "%.2f", score)
        case .keywordMatch:
            return String(format: "%.0f", score)
        }
    }

    private var caption: String {
        switch scale {
        case .similarity: return "similarity"
        case .keywordMatch: return "keyword score"
        }
    }

    var body: some View {
        Text(formatted)
            .font(.subheadline.monospacedDigit())
            .foregroundStyle(.secondary)
            .accessibilityLabel("\(caption) \(formatted)")
    }
}
