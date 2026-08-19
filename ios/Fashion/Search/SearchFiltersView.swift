import SwiftUI
import PhotosUI
import UIKit

/// The filter sheet.
///
/// A sheet rather than an always-visible filter bar: filters are secondary to
/// the photographs, and a permanent row of six controls above the results
/// would make chrome compete with content on every scroll. The button that
/// opens it names the active filters, so the state is never hidden.
struct SearchFiltersView: View {
    @Binding var filters: SearchFilters
    let colourVocab: [String]
    let categoryVocab: [String]

    @Environment(\.dismiss) private var dismiss
    @State private var colourPhotoItem: PhotosPickerItem?
    @State private var skinPhotoItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            Form {
                colourSection
                categorySection

                Section {
                    Toggle("US brands only", isOn: $filters.usOnly)
                    Toggle("Men’s clothing only", isOn: $filters.mensOnly)
                } footer: {
                    Text("Both filter the outfit corpus. The product catalog behind identification is already US men’s only, so these change which outfit photos come back, not which products exist.")
                }

                skinSection
            }
            .navigationTitle("Filters")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Clear") { filters = SearchFilters() }
                        .disabled(!filters.isActive)
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onChange(of: colourPhotoItem) { _, item in
                guard let item else { return }
                Task {
                    if let image = await PhotoLibraryLoader.image(from: item) {
                        filters.colourRGB = DominantColour.average(of: image)
                        filters.colourName = nil
                    }
                    colourPhotoItem = nil
                }
            }
            .onChange(of: skinPhotoItem) { _, item in
                guard let item else { return }
                Task {
                    filters.skinReferenceImage = await PhotoLibraryLoader.image(from: item)
                    skinPhotoItem = nil
                }
            }
        }
    }

    // MARK: Colour

    private var colourSection: some View {
        Section {
            Picker("Colour", selection: Binding(
                get: { filters.colourName ?? "" },
                set: { newValue in
                    filters.colourName = newValue.isEmpty ? nil : newValue
                    if !newValue.isEmpty { filters.colourRGB = nil }
                })) {
                Text("Any").tag("")
                ForEach(colourVocab, id: \.self) { colour in
                    Text(colour.capitalized).tag(colour)
                }
            }

            PhotosPicker(selection: $colourPhotoItem, matching: .images) {
                Label("Pick a colour from a photo", systemImage: "eyedropper")
            }
            .minimumHitTarget()

            if let rgb = filters.colourRGB {
                HStack(spacing: Theme.Space.snug) {
                    RoundedRectangle(cornerRadius: Theme.Radius.control / 2, style: .continuous)
                        .fill(DominantColour.swatch(rgb))
                        .frame(width: 28, height: 28)
                        .overlay(
                            RoundedRectangle(cornerRadius: Theme.Radius.control / 2, style: .continuous)
                                .strokeBorder(.separator))
                        .accessibilityHidden(true)
                    Text("Picked colour")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Button("Remove") { filters.colourRGB = nil }
                        .minimumHitTarget()
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("Picked colour: \(DominantColour.description(rgb))")
            }
        } header: {
            Text("Colour")
        } footer: {
            Text("The list is the set of colours the index actually knows. Picking from a photo sends the photo’s average colour, shown above as a swatch before it’s used.")
        }
    }

    private var categorySection: some View {
        Section {
            Picker("Garment type", selection: Binding(
                get: { filters.category ?? "" },
                set: { filters.category = $0.isEmpty ? nil : $0 })) {
                Text("Any").tag("")
                ForEach(categoryVocab, id: \.self) { category in
                    Text(category.capitalized).tag(category)
                }
            }
        } header: {
            Text("Garment type")
        } footer: {
            Text("Outfit photos are labelled automatically and never checked by a person. Layered pieces are often merged, so “outerwear” in particular is unreliable.")
        }
    }

    // MARK: Skin tone

    /// The most carefully worded control in the app.
    ///
    /// What it actually does: finds outfit photos whose skin *reads* similarly
    /// under similar lighting. What it is not: a measurement of anyone's skin
    /// tone, and not a position on a tone scale. Absolute tone binning was
    /// built, measured badly, and deliberately not shipped — so this control
    /// carries no tone names, no numeric readout and no swatches. The ends of
    /// the slider are described in terms of how skin *appears in a
    /// photograph*, which is the only claim the data supports.
    ///
    /// The photo option is listed first because it asks for an example rather
    /// than for a self-classification, which is both more accurate and a much
    /// better thing to ask a person for.
    private var skinSection: some View {
        Section {
            PhotosPicker(selection: $skinPhotoItem, matching: .images) {
                Label("Use a reference photo", systemImage: "person.crop.square")
            }
            .minimumHitTarget()

            if filters.skinReferenceImage != nil {
                HStack {
                    Text("Reference photo set")
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Button("Remove") { filters.skinReferenceImage = nil }
                        .minimumHitTarget()
                }
            }

            Toggle("Adjust by hand instead", isOn: Binding(
                get: { filters.skinTone != nil },
                set: { filters.skinTone = $0 ? 0.5 : nil }))
                .disabled(filters.skinReferenceImage != nil)

            if let value = filters.skinTone, filters.skinReferenceImage == nil {
                Slider(value: Binding(get: { value },
                                      set: { filters.skinTone = $0 }),
                       in: 0...1) {
                    Text("How skin appears in the photograph")
                } minimumValueLabel: {
                    Text("Lighter-reading").font(.caption2).foregroundStyle(.secondary)
                } maximumValueLabel: {
                    Text("Darker-reading").font(.caption2).foregroundStyle(.secondary)
                }
                .accessibilityLabel("How skin appears in the photograph")
                // Percentage along the control, not a tone value. There is no
                // scale here that a number could refer to.
                .accessibilityValue("\(Int(value * 100)) percent along the range")
            }
        } header: {
            Text("Skin in results")
        } footer: {
            Text("This is relative, not a tone scale. It finds outfit photos whose skin reads similarly to your reference under similar lighting — lighting affects it as much as skin does. Nothing about your skin tone is measured, categorised or stored.")
        }
    }
}

#Preview("Filters") {
    @Previewable @State var filters = SearchFilters()
    return SearchFiltersView(filters: $filters,
                             colourVocab: StubFashionAPI.colourVocab,
                             categoryVocab: StubFashionAPI.categoryVocab)
}
