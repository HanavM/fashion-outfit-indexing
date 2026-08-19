import SwiftUI
import PhotosUI
import UIKit

/// "See it on you" — the escape hatch when search disappoints.
///
/// The design is dominated by one requirement: **the output must be
/// unmistakably marked as generated.** That is done three ways, deliberately
/// redundantly, because a screenshot of this screen will outlive the screen:
///
/// 1. A permanent caption directly under the image, in body type — not a
///    footnote, not a tooltip, not something that scrolls away.
/// 2. The disclosure travels on `TryOnPreview` itself, so no view can render
///    the image without holding the words.
/// 3. There is no share or save button anywhere. The picture cannot leave the
///    app from here, which is also consistent with the rest of the product
///    having no export.
struct TryOnView: View {
    let outfit: OutfitResult

    @Environment(AppEnvironment.self) private var environment
    @Environment(\.dismiss) private var dismiss

    @State private var personItem: PhotosPickerItem?
    @State private var personImage: UIImage?
    @State private var preview: TryOnPreview?
    @State private var error: Error?
    @State private var isGenerating = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: Theme.Space.margin) {
                    if let preview {
                        generated(preview)
                    } else {
                        setup
                    }
                }
                .padding(.horizontal, Theme.Space.margin)
                .padding(.bottom, Theme.Space.section)
            }
            .background(Color(.systemBackground))
            .navigationTitle("See it on you")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .onChange(of: personItem) { _, item in
                guard let item else { return }
                Task {
                    personImage = await PhotoLibraryLoader.image(from: item)
                    personItem = nil
                }
            }
        }
    }

    @ViewBuilder
    private var setup: some View {
        Text("Use a photo of yourself to preview this look.")
            .font(.body)
            .fixedSize(horizontal: false, vertical: true)

        Qualifier("The result is made by an image model. It is not a photograph, it will not be accurate about fit or fabric, and it should not be treated as a picture of you wearing this.")

        if let personImage {
            Image(uiImage: personImage)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(maxHeight: 220)
                .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.photo, style: .continuous))
                .accessibilityLabel("The photo of you that will be used")
        }

        PhotosPicker(selection: $personItem, matching: .images) {
            Label(personImage == nil ? "Choose a photo of you" : "Choose a different photo",
                  systemImage: "person.crop.square")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.bordered)
        .minimumHitTarget()

        if !environment.tryOn.isConfigured {
            InlineErrorView(title: "No image service configured",
                            advice: "This feature needs an Azure AI Foundry endpoint, which hasn’t been set up yet. Everything else works without it.")
        } else if isGenerating {
            VStack(spacing: Theme.Space.snug) {
                ProgressView()
                Text("Generating")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, Theme.Space.section)
        } else {
            Button("Generate preview") { Task { await generate() } }
                .buttonStyle(.borderedProminent)
                .minimumHitTarget()
                .frame(maxWidth: .infinity)
                .disabled(personImage == nil)
        }

        if let error {
            InlineErrorView(title: (error as? TryOnError)?.errorDescription ?? error.localizedDescription,
                            advice: nil,
                            retry: personImage == nil ? nil : { Task { await generate() } })
        }
    }

    @ViewBuilder
    private func generated(_ preview: TryOnPreview) -> some View {
        Image(uiImage: preview.image)
            .resizable()
            .aspectRatio(contentMode: .fit)
            .frame(maxWidth: .infinity)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.photo, style: .continuous))
            .accessibilityLabel("Generated image. \(preview.disclosure)")

        // Body type, immediately under the image, not a caption in grey at
        // the bottom of the screen.
        Text(preview.disclosure)
            .font(.body.weight(.medium))
            .fixedSize(horizontal: false, vertical: true)

        Qualifier("Made \(preview.generatedAt.formatted(date: .abbreviated, time: .shortened)) from a photo you supplied and an outfit photo from \(outfit.source ?? "the corpus").")

        Button("Start over") {
            self.preview = nil
            self.personImage = nil
        }
        .buttonStyle(.bordered)
        .minimumHitTarget()
    }

    private func generate() async {
        guard let personImage, let data = ImageEncoding.prepared(personImage) else { return }
        isGenerating = true
        error = nil
        defer { isGenerating = false }
        do {
            preview = try await environment.tryOn.generatePreview(
                personImage: data, outfit: outfit, referenceImage: nil)
        } catch {
            self.error = error
        }
    }
}

#Preview("See it on you") {
    TryOnView(outfit: StubFashionAPI.outfitFixtures(matching: ["black jacket"])[0])
        .environment(AppEnvironment.preview)
}
