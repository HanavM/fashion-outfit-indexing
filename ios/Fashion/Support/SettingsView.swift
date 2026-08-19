import SwiftUI

/// Settings, which is mostly "where does the API key come from" plus the
/// standing disclosures about what this product is.
///
/// The disclosures live in a screen of their own rather than only inline,
/// because the inline versions are necessarily short. Someone who wants the
/// full story should be able to find it in one place.
struct SettingsView: View {
    @Environment(AppEnvironment.self) private var environment
    @Environment(\.dismiss) private var dismiss
    @State private var enteredKey: String = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("Source", value: environment.configuration.source.rawValue)
                    if environment.configuration.isConfigured {
                        Button("Remove stored key", role: .destructive) {
                            environment.configuration.clearStoredKey()
                            environment.reloadAPI()
                        }
                        .minimumHitTarget()
                    } else {
                        SecureField("Paste API key", text: $enteredKey)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        Button("Save to Keychain") {
                            environment.configuration.store(apiKey: enteredKey)
                            environment.reloadAPI()
                            enteredKey = ""
                        }
                        .disabled(enteredKey.trimmed.isEmpty)
                        .minimumHitTarget()
                    }
                } header: {
                    Text("API key")
                } footer: {
                    Text("Stored in the Keychain on this device. The key is never in the app’s source. Without one, the app shows demo data and labels it as such.")
                }

                Section("What this can and can’t do") {
                    disclosure(
                        "The catalog is small and specific",
                        environment.catalogDescription + " Photograph something outside that and it will still confidently name the closest thing it knows. That’s why identification always shows alternatives and a score."
                    )
                    disclosure(
                        "Garment labels are unchecked",
                        "The garments listed under an outfit photo come from a model, not a person. Roughly 9 in 10 are right, judged by eye on a small sample. Layered pieces — a jacket over a tee — are frequently merged into one."
                    )
                    disclosure(
                        "The skin-tone control is relative",
                        "It finds outfits whose skin reads similarly under similar lighting. It is not a tone scale and does not measure or record anyone’s skin tone. Absolute tone binning was tried, measured badly, and deliberately not shipped."
                    )
                    disclosure(
                        "These are photographs of real people",
                        "They were collected with a record of where each came from, but not with anyone’s permission. Every result links back to its original post, and the app has no sharing or export anywhere by design."
                    )
                    disclosure(
                        "“See it on you” output is generated",
                        "It is a machine-made picture, not a photograph of you and not a photograph of that outfit. It is labelled wherever it appears."
                    )
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func disclosure(_ title: String, _ body: String) -> some View {
        VStack(alignment: .leading, spacing: Theme.Space.tight) {
            Text(title).font(.subheadline.weight(.medium))
            Text(body)
                .font(.footnote)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, Theme.Space.hairline)
        .accessibilityElement(children: .combine)
    }
}

#Preview("Settings") {
    SettingsView().environment(AppEnvironment.preview)
}
