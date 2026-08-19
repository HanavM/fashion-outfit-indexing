import SwiftUI

/// A photograph of a person, filling its frame.
///
/// The content of this app is the photograph, so this view gives it the whole
/// rectangle and adds nothing: no border, no shadow, no gradient scrim, no
/// rounded-glass overlay. Chrome is elsewhere.
///
/// The missing-image state is a plain neutral fill with a word, not a
/// shimmering skeleton. A shimmer implies "arriving shortly", which is a
/// promise this view cannot keep — the corpus has dead links, and previews
/// have no images at all.
struct OutfitPhoto: View {
    let url: URL?
    /// Read out by VoiceOver in place of the image. Assembled from what the
    /// backend actually knows about the photo.
    let accessibilityDescription: String
    var contentMode: ContentMode = .fill

    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        Group {
            if let url {
                AsyncImage(url: url, transaction: Transaction(animation: Theme.Motion.content(reduceMotion: reduceMotion))) { phase in
                    switch phase {
                    case .success(let image):
                        image.resizable().aspectRatio(contentMode: contentMode)
                    case .failure:
                        unavailable("Photo unavailable")
                    case .empty:
                        placeholderSurface
                    @unknown default:
                        placeholderSurface
                    }
                }
            } else {
                unavailable("No photo")
            }
        }
        .background(placeholderSurface)
        .clipped()
        .accessibilityElement()
        .accessibilityLabel(accessibilityDescription)
    }

    private var placeholderSurface: some View {
        Rectangle().fill(.quaternary)
    }

    private func unavailable(_ text: String) -> some View {
        ZStack {
            placeholderSurface
            Text(text)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(Theme.Space.tight)
        }
    }
}

/// An empty state that says something true.
///
/// Every use of this passes a real explanation — how big the searched corpus
/// was, what the catalog covers — because "No results" on its own leaves the
/// user unable to tell a bad query from a broken app.
struct EmptyStateView: View {
    let title: String
    let explanation: String
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        VStack(spacing: Theme.Space.snug) {
            Text(title)
                .font(.headline)
                .multilineTextAlignment(.center)
            Text(explanation)
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if let actionTitle, let action {
                Button(actionTitle, action: action)
                    .buttonStyle(.bordered)
                    .minimumHitTarget()
                    .padding(.top, Theme.Space.tight)
            }
        }
        .padding(.horizontal, Theme.Space.section)
        .padding(.vertical, Theme.Space.section)
        .frame(maxWidth: .infinity)
    }
}

/// Shown whenever the screen's data came from fixtures.
struct DemoDataBanner: View {
    var body: some View {
        Text("Demo data — no API key configured. Add one in Settings.")
            .font(.footnote)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, Theme.Space.margin)
            .padding(.vertical, Theme.Space.tight)
            .background(.quaternary)
            .accessibilityLabel("Demo data. No API key configured. Add one in Settings.")
    }
}

/// The link back to where a photograph came from.
///
/// Present on every outfit, without exception. These are photographs of real
/// people collected with provenance but not permission; attribution back to
/// the original post is the minimum the product owes them, and it is why this
/// app has no sharing or export anywhere.
struct SourceLink: View {
    let outfit: OutfitResult

    var body: some View {
        if let postURL = outfit.postURL, let url = URL(string: postURL) {
            Link(destination: url) {
                HStack(spacing: Theme.Space.tight) {
                    Text(label)
                    Image(systemName: "arrow.up.right")
                        .font(.footnote)
                        .imageScale(.small)
                }
                .font(.subheadline)
            }
            .minimumHitTarget()
            .accessibilityLabel("Open the original post on \(outfit.source ?? "the source site")")
            .accessibilityHint("Opens in your browser")
        } else {
            Text("Original post unavailable")
                .font(.subheadline)
                .foregroundStyle(.secondary)
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
struct DetectedGarments: View {
    let categories: [String]
    var colors: [String] = []

    var body: some View {
        if !categories.isEmpty {
            VStack(alignment: .leading, spacing: Theme.Space.tight) {
                Text("Garments detected")
                    .font(.subheadline.weight(.medium))
                Text(descriptionText)
                    .font(.body)
                    .fixedSize(horizontal: false, vertical: true)
                Qualifier("Detected automatically and never checked by a person. Layered pieces are often merged into one.")
            }
            .accessibilityElement(children: .combine)
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

/// An error, rendered as a sentence rather than an alert.
///
/// Inline, because most failures here are transient or configuration issues
/// and a modal alert makes the user dismiss something before they can read
/// the screen behind it.
struct InlineErrorView: View {
    let title: String
    var advice: String?
    var retry: (() -> Void)?

    var body: some View {
        VStack(spacing: Theme.Space.snug) {
            Text(title)
                .font(.headline)
                .multilineTextAlignment(.center)
            if let advice {
                Text(advice)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let retry {
                Button("Try again", action: retry)
                    .buttonStyle(.bordered)
                    .minimumHitTarget()
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.horizontal, Theme.Space.section)
        .padding(.vertical, Theme.Space.section)
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
