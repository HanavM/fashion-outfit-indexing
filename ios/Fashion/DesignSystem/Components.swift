import SwiftUI

// MARK: - Photography

/// A photograph of a person, filling its frame.
///
/// The content of this app is the photograph, so this view gives it the whole
/// rectangle. Corners are 2pt: a print, not a bubble.
///
/// **The missing-image state is the interesting part.** The corpus has dead
/// links, the demo fixtures have no photographs at all, and a grey box with
/// the word "unavailable" made most of the app look broken. But the backend
/// *does* know the garment colours for every outfit — so when there is no
/// photograph, the tile renders those colours as stacked bands and names
/// them. It is honest (it never pretends to be the photo), it is useful (the
/// colours are real data), and it means a screen with no images still looks
/// like a fashion app rather than like a failed load.
///
/// Still no shimmer: a shimmer promises "arriving shortly", which is a promise
/// this view cannot keep.
struct OutfitPhoto: View {
    let url: URL?
    /// Read out by VoiceOver in place of the image. Assembled from what the
    /// backend actually knows about the photo.
    let accessibilityDescription: String
    var contentMode: ContentMode = .fill
    /// Garment colour words, used to draw the no-photograph state.
    var palette: [String] = []
    /// Suppresses the word overlay on very small tiles, where it would not
    /// fit. The colour bands still draw.
    var compact: Bool = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Group {
            if let url {
                // Not `AsyncImage`: the photo endpoint is authenticated and
                // `AsyncImage` cannot carry a header. See `AuthorizedImageLoader`.
                AuthorizedImage(url: url,
                                animation: Theme.Motion.content(reduceMotion: reduceMotion)) { phase in
                    switch phase {
                    case .success(let image):
                        image.resizable().aspectRatio(contentMode: contentMode)
                    case .failure:
                        unavailable("Photo unavailable")
                    case .empty:
                        ColourBands(names: palette)
                    @unknown default:
                        ColourBands(names: palette)
                    }
                }
            } else {
                unavailable("No photograph")
            }
        }
        .background(ColourBands(names: palette))
        .clipped()
        .accessibilityElement()
        .accessibilityLabel(accessibilityDescription)
    }

    private func unavailable(_ text: String) -> some View {
        ZStack(alignment: .bottomLeading) {
            ColourBands(names: palette)
            if !compact {
                VStack(alignment: .leading, spacing: Theme.Space.hairline) {
                    EyebrowText(text, tint: Theme.Palette.onInk)
                    if !palette.isEmpty {
                        // The colours are named, never only shown. This is the
                        // line that keeps the bands from being the sole
                        // carrier of the information.
                        Text(ListFormatter.localizedString(byJoining: palette))
                            .font(Theme.Typography.caption)
                            .foregroundStyle(Theme.Palette.onInk.opacity(0.85))
                            .lineLimit(2)
                    }
                }
                .padding(Theme.Space.snug)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.Palette.ink.opacity(0.55))
            }
        }
    }
}

/// The colour bands behind a missing photograph.
///
/// Vertical stripes in the garment's own colours, weighted so the first
/// colour dominates. Falls back to the plain surface fill when the outfit has
/// no colour data, which is the only case that still looks like a grey box —
/// and correctly so, because then there genuinely is nothing to show.
private struct ColourBands: View {
    let names: [String]

    private var colours: [Color] {
        names.compactMap { GarmentColour.swatch($0) }
    }

    var body: some View {
        GeometryReader { geometry in
            let resolved = colours
            if resolved.isEmpty {
                Theme.Palette.surface
            } else {
                HStack(spacing: 0) {
                    ForEach(Array(resolved.enumerated()), id: \.offset) { index, colour in
                        colour.frame(width: width(for: index,
                                                  count: resolved.count,
                                                  total: geometry.size.width))
                    }
                }
            }
        }
        .accessibilityHidden(true)
    }

    /// The first colour takes 60% when there is more than one, so the bands
    /// read as "mostly brown, some navy" rather than as an even flag.
    private func width(for index: Int, count: Int, total: CGFloat) -> CGFloat {
        guard count > 1 else { return total }
        let lead = total * 0.6
        return index == 0 ? lead : (total - lead) / CGFloat(count - 1)
    }
}

// MARK: - Colour

/// A garment colour as a swatch plus its name.
///
/// The pair is the component — there is no way to render the swatch without
/// the word, which is what keeps colour from being the only signal.
struct ColourTag: View {
    let name: String
    var size: CGFloat = 12

    var body: some View {
        HStack(spacing: Theme.Space.tight) {
            if let colour = GarmentColour.swatch(name) {
                Circle()
                    .fill(colour)
                    .frame(width: size, height: size)
                    .overlay(
                        Circle().strokeBorder(Theme.Palette.hairline,
                                              lineWidth: GarmentColour.needsOutline(name) ? 1 : 0))
                    .accessibilityHidden(true)
            }
            Text(name.capitalized)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Palette.inkSecondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(name) garment colour")
    }
}

/// Just the dots, for tiles too small for words.
///
/// Only ever used where the same colours are also spelled out in the tile's
/// `accessibilityLabel`, so the information is never lost.
struct ColourDots: View {
    let names: [String]
    var size: CGFloat = 8

    var body: some View {
        HStack(spacing: 3) {
            ForEach(Array(names.prefix(4).enumerated()), id: \.offset) { _, name in
                if let colour = GarmentColour.swatch(name) {
                    Circle()
                        .fill(colour)
                        .frame(width: size, height: size)
                        .overlay(
                            Circle().strokeBorder(Theme.Palette.hairline,
                                                  lineWidth: GarmentColour.needsOutline(name) ? 0.5 : 0))
                }
            }
        }
        .accessibilityHidden(true)
    }
}

// MARK: - Controls

/// A tag-shaped toggle. Pills, because they are tags.
///
/// Selected state changes fill, text colour **and** weight, so it survives
/// greyscale and Increase Contrast.
struct Chip: View {
    let title: String
    var swatch: Color?
    var outlineSwatch: Bool = false
    var isSelected: Bool = false
    var action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: Theme.Space.tight) {
                if let swatch {
                    Circle()
                        .fill(swatch)
                        .frame(width: 12, height: 12)
                        .overlay(
                            Circle().strokeBorder(
                                isSelected ? Theme.Palette.onAccent : Theme.Palette.hairline,
                                lineWidth: outlineSwatch ? 1 : 0))
                        .accessibilityHidden(true)
                }
                Text(title)
                    .font(.system(.subheadline, weight: isSelected ? .semibold : .regular))
            }
            .foregroundStyle(isSelected ? Theme.Palette.onAccent : Theme.Palette.ink)
            .padding(.horizontal, Theme.Space.snug + 2)
            .padding(.vertical, 9)
            .background(
                Capsule().fill(isSelected ? Theme.Palette.accent : Theme.Palette.surface))
            .overlay(
                Capsule().strokeBorder(isSelected ? .clear : Theme.Palette.hairline, lineWidth: 1))
        }
        .buttonStyle(.plain)
        .frame(minHeight: Theme.minimumTouchTarget)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}

/// The app's primary button: a solid rust slab with square-ish corners.
///
/// Text uses `onAccent`, never `ink` — ink on accent is 3.0:1 and fails AA.
struct SlabButtonStyle: ButtonStyle {
    var tint: Color = Theme.Palette.accent
    var foreground: Color = Theme.Palette.onAccent

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.subheadline, weight: .semibold))
            .tracking(0.6)
            .foregroundStyle(foreground)
            .padding(.horizontal, Theme.Space.margin)
            .frame(minHeight: Theme.minimumTouchTarget)
            .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .fill(tint))
            .opacity(configuration.isPressed ? 0.82 : 1)
    }
}

/// The secondary button: an outlined slab, same geometry.
struct OutlineButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(.subheadline, weight: .semibold))
            .tracking(0.6)
            .foregroundStyle(Theme.Palette.ink)
            .padding(.horizontal, Theme.Space.margin)
            .frame(minHeight: Theme.minimumTouchTarget)
            .background(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .strokeBorder(Theme.Palette.ink, lineWidth: 1.5))
            .opacity(configuration.isPressed ? 0.6 : 1)
    }
}

// MARK: - Honesty surfaces

/// A short piece of text that qualifies the confidence of what is next to it.
///
/// This exists because the product has real, measured limits that must not be
/// designed away: the catalog is 20 US men's brands and will confidently name
/// the closest thing it knows to anything else; the garment labels on outfit
/// photos are unvalidated model output. Those facts belong beside the claim,
/// every time, so this is a component rather than an ad-hoc `Text`.
///
/// It now carries a **pine rule down its left edge**. That is a deliberate
/// upgrade in prominence, not a decoration: the previous version was grey
/// footnote text that read as boilerplate and got skipped. Pine rather than
/// rust, so a caveat is never confused with something you can press, and
/// rather than red, so it reads as context rather than as an error.
struct Qualifier: View {
    private let text: String

    init(_ text: String) {
        self.text = text
    }

    var body: some View {
        HStack(alignment: .top, spacing: Theme.Space.snug) {
            Rectangle()
                .fill(Theme.Palette.secondaryAccent)
                .frame(width: 2)
                .accessibilityHidden(true)
            Text(text)
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Palette.inkSecondary)
                // Never truncate a caveat. It wraps as far as it needs to.
                .fixedSize(horizontal: false, vertical: true)
        }
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(text)
    }
}

/// Shown whenever the screen's data came from fixtures.
///
/// Redesigned from a flat grey strip into a rust-ruled band with a tracked
/// eyebrow. It is *more* visible than before, which is the right direction
/// for a disclosure: it now looks like a masthead standfirst rather than like
/// a disabled toolbar.
struct DemoDataBanner: View {
    var body: some View {
        HStack(alignment: .center, spacing: Theme.Space.snug) {
            Rectangle()
                .fill(Theme.Palette.accent)
                .frame(width: 3)
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                EyebrowText("Demo data", tint: Theme.Palette.accent)
                Text("No API key configured. Add one in Settings.")
                    .font(Theme.Typography.caption)
                    .foregroundStyle(Theme.Palette.inkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .frame(minHeight: 40)
        .padding(.trailing, Theme.Space.margin)
        .padding(.vertical, Theme.Space.tight)
        .background(Theme.Palette.surface)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Demo data. No API key configured. Add one in Settings.")
    }
}

/// The link back to where a photograph came from.
///
/// Present on every outfit, without exception. These are photographs of real
/// people collected with provenance but not permission; attribution back to
/// the original post is the minimum the product owes them, and it is why this
/// app has no sharing or export anywhere.
///
/// Rendered in pine on a surface pill so it reads as a real destination
/// rather than as a footnote — attribution should look like something you are
/// invited to follow.
struct SourceLink: View {
    let outfit: OutfitResult

    var body: some View {
        if let postURL = outfit.postURL, let url = URL(string: postURL) {
            Link(destination: url) {
                HStack(spacing: Theme.Space.tight) {
                    Text(label)
                        .font(.system(.footnote, weight: .semibold))
                        .multilineTextAlignment(.leading)
                    Image(systemName: "arrow.up.right")
                        .font(.system(.caption2, weight: .bold))
                }
                .foregroundStyle(Theme.Palette.secondaryAccent)
                .padding(.horizontal, Theme.Space.snug)
                .padding(.vertical, 8)
                .background(Capsule().fill(Theme.Palette.surface))
                .overlay(Capsule().strokeBorder(Theme.Palette.secondaryAccent.opacity(0.35),
                                                lineWidth: 1))
            }
            .frame(minHeight: Theme.minimumTouchTarget)
            .accessibilityLabel("Open the original post on \(outfit.source ?? "the source site")")
            .accessibilityHint("Opens in your browser")
        } else {
            Text("Original post unavailable")
                .font(Theme.Typography.caption)
                .foregroundStyle(Theme.Palette.inkSecondary)
        }
    }

    private var label: String {
        if let author = outfit.author, !author.isEmpty {
            return "Original post by \(author)"
        }
        if let source = outfit.source, !source.isEmpty {
            return "Original post on \(source.capitalized)"
        }
        return "Original post"
    }
}

/// The garments a model thinks are in a photo, with the caveat attached.
///
/// The qualifier is part of the component rather than the caller's
/// responsibility, so there is no way to render these labels without it.
/// The detections are ~91% precise by eyeball on 40 photos — the same 40 the
/// threshold was picked on — and `outerwear` in particular sits near 41%
/// agreement for a structural reason that better data will not fix.
///
/// Categories are now chips and colours are now swatches-with-names. Both are
/// non-interactive here; they are typeset as tags because that is what they
/// are, and the colour is the one piece of real colour in the payload.
struct DetectedGarments: View {
    let categories: [String]
    var colors: [String] = []

    var body: some View {
        if !categories.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Space.snug) {
                EyebrowText("Garments detected")

                FlowRow(spacing: Theme.Space.tight) {
                    ForEach(categories, id: \.self) { category in
                        Text(category.capitalized)
                            .font(.system(.subheadline, weight: .medium))
                            .foregroundStyle(Theme.Palette.ink)
                            .padding(.horizontal, Theme.Space.snug)
                            .padding(.vertical, 6)
                            .background(Capsule().fill(Theme.Palette.surface))
                            .overlay(Capsule().strokeBorder(Theme.Palette.hairline, lineWidth: 1))
                    }
                }

                if !colors.isEmpty {
                    FlowRow(spacing: Theme.Space.snug) {
                        ForEach(colors, id: \.self) { ColourTag(name: $0) }
                    }
                }

                Qualifier("Detected automatically and never checked by a person. Layered pieces are often merged into one.")
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel("Garments detected, automatically and unverified: \(descriptionText)")
        }
    }

    private var descriptionText: String {
        var text = ListFormatter.localizedString(byJoining: categories)
        if !colors.isEmpty {
            text += ". Colours: " + ListFormatter.localizedString(byJoining: colors)
        }
        return text
    }
}

// MARK: - States

/// An empty state that says something true.
///
/// Every use of this passes a real explanation — how big the searched corpus
/// was, what the catalog covers — because "No results" on its own leaves the
/// user unable to tell a bad query from a broken app.
///
/// **Left-aligned, not centred.** Centred grey text in the middle of a white
/// screen was the single biggest reason the app read as unfinished. Aligned
/// to the same column as everything else, under a display-sized headline, it
/// reads as a page with something to say.
struct EmptyStateView: View {
    let title: String
    let explanation: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.snug) {
            Rectangle()
                .fill(Theme.Palette.accent)
                .frame(width: 44, height: 3)
                .accessibilityHidden(true)

            DisplayText(title, size: 30)

            Text(explanation)
                .font(Theme.Typography.body)
                .foregroundStyle(Theme.Palette.inkSecondary)
                .fixedSize(horizontal: false, vertical: true)

            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(SlabButtonStyle())
                    .padding(.top, Theme.Space.tight)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Theme.Space.margin)
        .padding(.vertical, Theme.Space.section)
    }
}

/// An error, rendered as a sentence rather than an alert.
///
/// Inline, because most failures here are transient or configuration issues
/// and a modal alert makes the user dismiss something before they can read
/// the screen behind it.
///
/// The rust rule and eyebrow are the only place the accent is used to mean
/// "something went wrong", and the word "Problem" carries that meaning too,
/// so the colour is not doing it alone.
struct InlineErrorView: View {
    let title: String
    var advice: String?
    var retry: (() -> Void)?

    var body: some View {
        VStack(alignment: .leading, spacing: Theme.Space.snug) {
            EyebrowText("Problem", tint: Theme.Palette.accent)

            Text(title)
                .font(Theme.Typography.title)
                .foregroundStyle(Theme.Palette.ink)
                .fixedSize(horizontal: false, vertical: true)

            if let advice {
                Text(advice)
                    .font(Theme.Typography.bodySmall)
                    .foregroundStyle(Theme.Palette.inkSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let retry {
                Button("Try again", action: retry)
                    .buttonStyle(OutlineButtonStyle())
                    .padding(.top, Theme.Space.tight)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(Theme.Space.margin)
        .background(
            RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous)
                .fill(Theme.Palette.surface))
        .overlay(alignment: .leading) {
            Rectangle().fill(Theme.Palette.accent).frame(width: 3)
                .accessibilityHidden(true)
        }
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.control, style: .continuous))
        .padding(.horizontal, Theme.Space.margin)
    }
}

/// A match score, shown as a number rather than as a bar or a badge.
///
/// Deliberately plain, and deliberately **not** coloured. A five-star widget,
/// a progress bar or a green/amber/red pill would imply a calibration this
/// model does not have — the wrong answer has outscored the right one 0.922
/// to 0.854 in this system's own evaluation. This is the one place in the
/// redesign where colour was available and refused: encoding a meaningless
/// ordering in hue would be the most misleading thing the app could do.
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
            .font(Theme.Typography.figure)
            .foregroundStyle(Theme.Palette.inkSecondary)
            .accessibilityLabel("\(caption) \(formatted)")
    }
}

// MARK: - Layout

/// A wrapping row. Needed because chips are variable width and a fixed
/// `HStack` truncates them at large Dynamic Type sizes.
///
/// Written by hand rather than adding a package — the app has zero
/// third-party dependencies and this is thirty lines.
struct FlowRow: Layout {
    var spacing: CGFloat = 8

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0 && x + size.width > maxWidth {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
        return CGSize(width: maxWidth == .infinity ? x : maxWidth, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize,
                       subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX && x + size.width > bounds.maxX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), anchor: .topLeading,
                          proposal: ProposedViewSize(size))
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}

extension View {
    /// Applies an error's own words. Keeps the mapping in one place.
    @ViewBuilder
    func replacingContent(with error: Error?, retry: (() -> Void)? = nil) -> some View {
        if let error {
            let fashionError = error as? FashionAPIError
            InlineErrorView(title: fashionError?.errorDescription ?? error.localizedDescription,
                            advice: fashionError?.recoveryAdvice,
                            retry: retry)
        } else {
            self
        }
    }
}
