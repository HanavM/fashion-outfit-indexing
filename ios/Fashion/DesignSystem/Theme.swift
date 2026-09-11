import SwiftUI

/// The whole visual vocabulary of the app, in one file.
///
/// The register is **editorial**: this is a fashion product, so it is allowed
/// to look like one. The rules it encodes:
///
/// - **Photographs are the content and they are printed square.** Images get
///   the full column, or the full bleed, with a 2pt radius — a print, not a
///   bubble. The uniform 12pt-rounded-everything look is deliberately absent.
/// - **Three radii, each meaning something.** Photos are near-square, chips
///   are pills because they are tags, buttons are 4pt because they are
///   buttons. A shape change signals a kind change.
/// - **A real palette, in the asset catalogue, with light and dark and
///   increased-contrast variants.** Warm paper and warm ink rather than
///   system white and system grey; rust as the primary; pine as the
///   supporting colour. Every text pairing clears WCAG AA — see the ratios in
///   `Palette` below.
/// - **Type carries the hierarchy loudly.** Compressed black display faces
///   against small tracked small-caps labels. Everything routes through
///   `Theme.Typography`, and all of it is Dynamic Type native.
/// - **Colour is never the only carrier of meaning.** Swatches carry names,
///   selected states carry weight and shape changes, the honesty surfaces
///   carry words. Colour is added on top of a signal that already works.
enum Theme {

    /// The palette. Names are roles, not hues, so a view reads as its intent.
    ///
    /// Measured contrast against `paper` (light / dark):
    /// - `ink` 16.5 / 16.7
    /// - `inkSecondary` 6.9 / 7.1
    /// - `accent` 5.6 / 7.3
    /// - `secondaryAccent` 7.0 / 8.7
    ///
    /// Against `surface` the lowest is `accent` at 5.0 / 6.5. All clear AA
    /// for body text, so no pairing in here is restricted to large text.
    ///
    /// The one pairing that does **not** pass is ink on accent (3.0), which
    /// is why anything sitting on a solid accent fill uses `onAccent`
    /// (5.6 / 7.3) and never `ink`.
    enum Palette {
        /// App background. Warm off-white, not `systemBackground`.
        static let paper = Color("Paper")
        /// Filled surfaces: chips, tiles, wells, image placeholders.
        static let surface = Color("Surface")
        /// Primary text, and the fill for hero surfaces that invert.
        static let ink = Color("Ink")
        /// Supporting text. Warm, not a neutral system grey.
        static let inkSecondary = Color("InkSecondary")
        /// Rules and edges. Visible, unlike `.separator`.
        static let hairline = Color("Hairline")
        /// Rust. Actions, active state, the app's signature.
        static let accent = Color("AccentColor")
        /// Pine. The supporting colour — provenance, confirmations, honesty
        /// surfaces. Distinct from accent so "this is a caveat" and "this is
        /// a thing you can press" never read the same.
        static let secondaryAccent = Color("SecondaryAccent")
        /// Text and glyphs on a solid `accent` fill.
        static let onAccent = Color("OnAccent")
        /// Text and glyphs on a solid `ink` fill.
        static let onInk = Color("OnInk")
    }

    /// A 4pt-derived spacing scale. Named by intent, not by number.
    ///
    /// Deliberately *not* applied uniformly. Photographs run to the bleed,
    /// text runs to `margin`, and chip rows run past the edge — the differing
    /// insets are what give a screen rhythm instead of a stack of identical
    /// cards.
    enum Space {
        /// Between a label and the thing it labels.
        static let hairline: CGFloat = 2
        /// Within a single unit of content.
        static let tight: CGFloat = 6
        /// The default gap between related elements.
        static let snug: CGFloat = 12
        /// Standard text inset. Wider than the system's 16 — an editorial
        /// column sits in from the edge further than a settings list does.
        static let margin: CGFloat = 20
        /// Between distinct groups.
        static let section: CGFloat = 36
    }

    /// Three radii, three meanings. See the type comment.
    enum Radius {
        /// Photographs. A print has square corners.
        static let photo: CGFloat = 2
        /// Buttons and wells.
        static let control: CGFloat = 4
        /// Chips and tags, which are pills because they are tags.
        static let chip: CGFloat = 999
    }

    /// Apple's minimum comfortable hit target. Any control smaller than this
    /// visually must still claim this much space.
    static let minimumTouchTarget: CGFloat = 44

    /// Type.
    ///
    /// The range here is intentionally wide — a 40pt compressed black display
    /// against a 11pt tracked caption is most of what makes the app read as
    /// editorial rather than as a form. Everything is built from system text
    /// styles, so Dynamic Type scales all of it; the display sizes use
    /// `@ScaledMetric` (see `DisplayText`) for the same reason.
    enum Typography {
        /// Section mastheads. Compressed and black — a magazine header.
        static func display(_ size: CGFloat) -> Font {
            .system(size: size, weight: .black).width(.compressed)
        }
        /// Headline inside content.
        static let title = Font.system(.title2, weight: .bold)
        /// Sub-headers within a screen.
        static let subtitle = Font.system(.headline, weight: .semibold)
        /// Body copy.
        static let body = Font.system(.body)
        /// Secondary body copy.
        static let bodySmall = Font.system(.subheadline)
        /// Small tracked uppercase labels. The counterweight to the display
        /// face, and the app's second most-used voice.
        static let label = Font.system(.caption, weight: .semibold)
        /// Caveats and provenance.
        static let caption = Font.system(.footnote)
        /// Numbers that should line up.
        static let figure = Font.system(.subheadline, weight: .medium).monospacedDigit()
    }

    /// Motion.
    ///
    /// This is a utility app wearing an editorial coat. Motion exists to
    /// explain what moved where and nothing else — no cinematic reveals, no
    /// staged cascades. Springs are short (<= 300ms) and interruptible, and
    /// every one of them is routed through `Motion.standard(reduceMotion:)`
    /// so that Reduce Motion collapses it to no animation at all.
    enum Motion {
        static func standard(reduceMotion: Bool) -> Animation? {
            reduceMotion ? nil : .spring(response: 0.28, dampingFraction: 0.9)
        }

        /// For changes that swap content wholesale (a results grid arriving).
        static func content(reduceMotion: Bool) -> Animation? {
            reduceMotion ? nil : .spring(response: 0.3, dampingFraction: 0.95)
        }
    }
}

// MARK: - Type components

/// A masthead. Large, compressed, black, and scaled by Dynamic Type.
///
/// `@ScaledMetric` rather than a fixed size, so the display face grows with
/// the user's text setting like every other piece of type in the app. It
/// wraps rather than truncating.
struct DisplayText: View {
    private let text: String
    private let base: CGFloat
    @ScaledMetric(relativeTo: .largeTitle) private var scale: CGFloat = 1

    init(_ text: String, size: CGFloat = 40) {
        self.text = text
        self.base = size
    }

    var body: some View {
        Text(text)
            .font(Theme.Typography.display(base * scale))
            .tracking(-0.5)
            .foregroundStyle(Theme.Palette.ink)
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// A small tracked uppercase label — the app's quiet voice.
///
/// Uppercasing happens here rather than in the string so that VoiceOver still
/// reads the original casing; `accessibilityLabel` restores it explicitly.
struct EyebrowText: View {
    private let text: String
    private var tint: Color

    init(_ text: String, tint: Color = Theme.Palette.inkSecondary) {
        self.text = text
        self.tint = tint
    }

    var body: some View {
        Text(text.uppercased())
            .font(Theme.Typography.label)
            .tracking(1.2)
            .foregroundStyle(tint)
            .accessibilityLabel(text)
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

    /// The app's page background. One call so no screen forgets.
    func paperBackground() -> some View {
        background(Theme.Palette.paper.ignoresSafeArea())
    }
}

// MARK: - Garment colour

/// Real colour for the colour words the index actually uses.
///
/// The backend's `colour_vocab` is a fixed list of ordinary colour names, and
/// rendering them as grey chips throws away the one piece of genuinely
/// colourful data the app has. So they become swatches.
///
/// Two rules keep this honest and accessible:
/// - A swatch is **never alone**. Every use pairs it with the colour's name,
///   because a swatch is unreadable to anyone who can't distinguish it and
///   invisible to VoiceOver.
/// - The hues are approximations of a *word*, not measurements of a garment,
///   so they are only ever used decoratively beside that word — never to
///   claim "this jacket is this colour".
enum GarmentColour {
    private static let table: [String: (Double, Double, Double)] = [
        "black":    (0.09, 0.09, 0.09),
        "white":    (0.97, 0.96, 0.94),
        "grey":     (0.58, 0.57, 0.55),
        "gray":     (0.58, 0.57, 0.55),
        "navy":     (0.11, 0.16, 0.31),
        "blue":     (0.20, 0.38, 0.65),
        "brown":    (0.42, 0.28, 0.17),
        "beige":    (0.83, 0.75, 0.62),
        "green":    (0.20, 0.45, 0.27),
        "olive":    (0.40, 0.42, 0.22),
        "red":      (0.72, 0.16, 0.13),
        "burgundy": (0.40, 0.11, 0.17),
        "orange":   (0.85, 0.44, 0.13),
        "yellow":   (0.90, 0.75, 0.20),
        "purple":   (0.38, 0.24, 0.49),
        "pink":     (0.87, 0.62, 0.66),
    ]

    /// `nil` for a word the vocabulary doesn't cover, so callers can fall
    /// back to text alone rather than inventing a hue.
    static func swatch(_ name: String) -> Color? {
        guard let rgb = table[name.lowercased().trimmed] else { return nil }
        return Color(.sRGB, red: rgb.0, green: rgb.1, blue: rgb.2)
    }

    /// Whether a swatch needs an outline to be visible against paper. White
    /// and near-white garments would otherwise disappear.
    static func needsOutline(_ name: String) -> Bool {
        ["white", "beige", "yellow"].contains(name.lowercased().trimmed)
    }
}
