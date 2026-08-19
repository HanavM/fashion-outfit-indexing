import SwiftUI
import UIKit

/// What happens between taking a photo and it becoming a closet item.
///
/// The design decision that matters here: **this screen never presents a
/// single answer.** `/identify` returns a ranked list, and on this backend
/// the top of that list is regularly wrong — verified on the live service,
/// where the correct product came back third behind a different sweatshirt
/// and a pair of trousers, separated by 0.019 of similarity. So the user is
/// shown the candidates, with scores, and asked to pick. "None of these"
/// is a first-class outcome, not a hidden escape hatch, because the catalog
/// is 20 US men's brands and most of the world's clothing is outside it.
struct CaptureReviewView: View {
    let image: UIImage

    @Environment(AppEnvironment.self) private var environment
    @Environment(\.dismiss) private var dismiss
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    @State private var response: IdentifyResponse?
    @State private var error: Error?
    @State private var isLoading = false
    @State private var clock = LoadingClock()
    @State private var selection: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.margin) {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxHeight: 260)
                        .frame(maxWidth: .infinity)
                        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.photo,
                                                    style: .continuous))
                        .accessibilityLabel("The photo you just took")

                    if isLoading {
                        LoadingView(clock: clock, label: "Matching against the catalog")
                    } else if let error {
                        InlineErrorView(
                            title: (error as? FashionAPIError)?.errorDescription ?? error.localizedDescription,
                            advice: (error as? FashionAPIError)?.recoveryAdvice,
                            retry: { Task { await identify() } })
                        saveAnywayButton
                    } else if let response {
                        results(for: response)
                    }
                }
                .padding(.horizontal, Theme.Space.margin)
                .padding(.bottom, Theme.Space.section)
            }
            .background(Color(.systemBackground))
            .navigationTitle("Add to closet")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .disabled(isLoading)
                }
            }
        }
        .task { await identify() }
    }

    @ViewBuilder
    private func results(for response: IdentifyResponse) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.snug) {

            // The gate fires before identity does. If the photo doesn't look
            // like clothing, saying so is far more useful than ranking five
            // sweatshirts against a photo of a chair.
            if response.garmentGate?.looksLikeClothing == false {
                VStack(alignment: .leading, spacing: Theme.Space.tight) {
                    Text("This doesn’t look like clothing")
                        .font(.headline)
                    Qualifier("You can still save it — the matches below are unlikely to mean anything.")
                }
            }

            Text("Which one is it?")
                .font(.headline)

            Qualifier("Matched against \(environment.catalogDescription) If your item isn’t from one of those brands, the closest match will still be listed — it just won’t be right.")

            VStack(spacing: 0) {
                ForEach(response.results) { candidate in
                    CandidateRow(candidate: candidate,
                                 isSelected: selection == candidate.productCode) {
                        withAnimation(Theme.Motion.standard(reduceMotion: reduceMotion)) {
                            selection = candidate.productCode
                        }
                    }
                    Divider()
                }

                CandidateRow.noneOfThese(isSelected: selection == nil) {
                    withAnimation(Theme.Motion.standard(reduceMotion: reduceMotion)) {
                        selection = nil
                    }
                }
            }

            if response.rejectThresholdCalibrated == false {
                Qualifier("This system can’t currently say “that isn’t in the catalog” on its own — the threshold that would do it is uncalibrated. It always returns its closest guesses.")
            }
        }
    }

    private var saveAnywayButton: some View {
        Button("Save without identifying") { save() }
            .buttonStyle(.bordered)
            .minimumHitTarget()
            .frame(maxWidth: .infinity)
    }

    private func identify() async {
        guard let data = ImageEncoding.prepared(image) else {
            error = FashionAPIError.transport("That photo couldn’t be read.")
            return
        }
        isLoading = true
        error = nil
        clock.start()
        defer {
            isLoading = false
            clock.stop()
        }
        do {
            let result = try await environment.api.identify(imageData: data, topK: 5)
            response = result
            // Nothing is pre-selected. Pre-ticking rank 1 would put the
            // system's guess in the user's mouth, and rank 1 is wrong often
            // enough here that the tap is worth asking for.
            selection = nil
        } catch {
            self.error = error
        }
    }

    private func save() {
        guard let item = environment.closet.add(image: image,
                                                candidates: response?.results ?? []) else {
            dismiss()
            return
        }
        if let selection {
            environment.closet.confirm(item, productCode: selection)
        } else if response != nil {
            environment.closet.rejectAllCandidates(for: item)
        }
        dismiss()
    }
}

/// One candidate, as a row.
///
/// Selection is a checkmark in the accent colour plus a weight change on the
/// title. Colour alone is never the signal — that would fail for anyone who
/// can't distinguish it, and the accent is the only colour this app spends.
private struct CandidateRow: View {
    let title: String
    let subtitle: String?
    let score: Double?
    let isSelected: Bool
    let action: () -> Void

    init(candidate: IdentifiedProduct, isSelected: Bool, action: @escaping () -> Void) {
        self.title = candidate.displayName
        self.subtitle = candidate.brand
        self.score = candidate.score
        self.isSelected = isSelected
        self.action = action
    }

    private init(title: String, subtitle: String?, score: Double?,
                 isSelected: Bool, action: @escaping () -> Void) {
        self.title = title
        self.subtitle = subtitle
        self.score = score
        self.isSelected = isSelected
        self.action = action
    }

    static func noneOfThese(isSelected: Bool, action: @escaping () -> Void) -> CandidateRow {
        CandidateRow(title: "None of these",
                     subtitle: "Keep it in the closet without a catalog match",
                     score: nil,
                     isSelected: isSelected,
                     action: action)
    }

    var body: some View {
        Button(action: action) {
            HStack(alignment: .firstTextBaseline, spacing: Theme.Space.snug) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? AnyShapeStyle(Color.accentColor)
                                                : AnyShapeStyle(.tertiary))
                    .imageScale(.large)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: Theme.Space.hairline) {
                    Text(title)
                        .font(.body.weight(isSelected ? .semibold : .regular))
                        .fixedSize(horizontal: false, vertical: true)
                    if let subtitle {
                        Text(subtitle)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if let score {
                    ScoreLabel(score)
                }
            }
            .padding(.vertical, Theme.Space.snug)
            .contentShape(Rectangle())
            .frame(minHeight: Theme.minimumTouchTarget)
        }
        .buttonStyle(.plain)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(accessibilityLabel)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private var accessibilityLabel: String {
        var parts = [title]
        if let subtitle { parts.append(subtitle) }
        if let score { parts.append("similarity \(String(format: "%.2f", score))") }
        return parts.joined(separator: ", ")
    }
}

#Preview("Capture review") {
    CaptureReviewView(image: UIImage(systemName: "tshirt") ?? UIImage())
        .environment(AppEnvironment.preview)
}
