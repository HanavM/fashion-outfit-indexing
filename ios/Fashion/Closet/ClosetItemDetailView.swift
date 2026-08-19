import SwiftUI

/// One garment you own.
///
/// Two jobs: let you correct what the system thinks this is, and let you ask
/// the product's actual question — "how does this look with my X" — which
/// hands the garment's photo to Search as the image part of a query.
struct ClosetItemDetailView: View {
    let item: ClosetItem

    @Environment(AppEnvironment.self) private var environment
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var nickname: String = ""
    @State private var isEditingNickname = false
    @State private var showingAlternatives = false
    @State private var confirmingDelete = false

    private var current: ClosetItem {
        environment.closet.items.first { $0.id == item.id } ?? item
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: Theme.Space.margin) {
                photo

                VStack(alignment: .leading, spacing: Theme.Space.tight) {
                    Text(current.displayTitle)
                        .font(.title2.weight(.semibold))
                        .fixedSize(horizontal: false, vertical: true)
                    if let brand = current.displayBrand {
                        Text(brand)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                identification

                pairingAction

                Divider()

                management
            }
            .padding(.horizontal, Theme.Space.margin)
            .padding(.bottom, Theme.Space.section)
        }
        .background(Color(.systemBackground))
        .navigationTitle("Item")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { nickname = current.nickname ?? "" }
    }

    private var photo: some View {
        Group {
            if let image = environment.closet.image(for: current) {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                ZStack {
                    Rectangle().fill(.quaternary)
                    Text("Photo unavailable")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(height: 240)
            }
        }
        .frame(maxWidth: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Radius.photo, style: .continuous))
        .accessibilityLabel("Photo of \(current.displayTitle)")
    }

    @ViewBuilder
    private var identification: some View {
        VStack(alignment: .leading, spacing: Theme.Space.snug) {
            if current.rejectedAllCandidates {
                Text("Not in the catalog")
                    .font(.headline)
                Qualifier("You said none of the suggested products matched. It stays in your closet as your own item.")
            } else if let confirmed = current.confirmed {
                HStack(alignment: .firstTextBaseline) {
                    Text("You confirmed this match")
                        .font(.headline)
                    Spacer()
                    ScoreLabel(confirmed.score)
                }
            } else if !current.candidates.isEmpty {
                Text("Unconfirmed guess")
                    .font(.headline)
                Qualifier("Nobody has confirmed this. The catalog’s top answer is often not the right one — the score is a similarity, not a probability that it’s correct.")
            }

            if !current.candidates.isEmpty {
                Button {
                    withAnimation(Theme.Motion.standard(reduceMotion: reduceMotion)) {
                        showingAlternatives.toggle()
                    }
                } label: {
                    Label(showingAlternatives ? "Hide other matches" : "See other matches",
                          systemImage: showingAlternatives ? "chevron.up" : "chevron.down")
                        .font(.subheadline)
                }
                .minimumHitTarget()

                if showingAlternatives {
                    VStack(spacing: 0) {
                        ForEach(current.candidates) { candidate in
                            Button {
                                environment.closet.confirm(current, productCode: candidate.productCode)
                            } label: {
                                HStack(alignment: .firstTextBaseline, spacing: Theme.Space.snug) {
                                    VStack(alignment: .leading, spacing: Theme.Space.hairline) {
                                        Text(candidate.displayName)
                                            .font(.body)
                                            .fixedSize(horizontal: false, vertical: true)
                                        if let brand = candidate.brand {
                                            Text(brand)
                                                .font(.footnote)
                                                .foregroundStyle(.secondary)
                                        }
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    ScoreLabel(candidate.score)
                                    if current.confirmedProductCode == candidate.productCode {
                                        Image(systemName: "checkmark")
                                            .foregroundStyle(Color.accentColor)
                                            .accessibilityHidden(true)
                                    }
                                }
                                .padding(.vertical, Theme.Space.snug)
                                .contentShape(Rectangle())
                                .frame(minHeight: Theme.minimumTouchTarget)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel("\(candidate.displayName), similarity \(String(format: "%.2f", candidate.score))")
                            .accessibilityHint("Confirms this as the match")
                            Divider()
                        }

                        Button("None of these are right") {
                            environment.closet.rejectAllCandidates(for: current)
                        }
                        .font(.body)
                        .padding(.vertical, Theme.Space.snug)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .minimumHitTarget()
                    }
                    .transition(.opacity)
                }
            }
        }
    }

    /// The product's real question.
    private var pairingAction: some View {
        Button {
            if let image = environment.closet.image(for: current) {
                environment.router.selectedTab = .search
                SearchComposition.shared.stage(image: image,
                                               caption: current.displayTitle)
            }
        } label: {
            Label("See how this looks with…", systemImage: "magnifyingglass")
                .frame(maxWidth: .infinity)
        }
        .buttonStyle(.borderedProminent)
        .minimumHitTarget()
        .accessibilityHint("Starts a search using this garment’s photo, so you can add what it should be worn with")
    }

    private var management: some View {
        VStack(alignment: .leading, spacing: Theme.Space.snug) {
            HStack {
                TextField("Call it something", text: $nickname)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { environment.closet.rename(current, to: nickname) }
                Button("Save") { environment.closet.rename(current, to: nickname) }
                    .buttonStyle(.bordered)
                    .minimumHitTarget()
                    .disabled(nickname == (current.nickname ?? ""))
            }

            Button("Remove from closet", role: .destructive) {
                confirmingDelete = true
            }
            .minimumHitTarget()
            .confirmationDialog("Remove this item?", isPresented: $confirmingDelete) {
                Button("Remove", role: .destructive) {
                    environment.closet.delete(current)
                }
            } message: {
                Text("The photo is deleted from this device.")
            }
        }
    }
}

#Preview("Closet item") {
    let environment = AppEnvironment.preview
    return NavigationStack {
        ClosetItemDetailView(item: environment.closet.items[1])
    }
    .environment(environment)
}
