import SwiftUI

@main
struct FashionApp: App {
    @State private var environment = AppEnvironment()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(environment)
                .task { await environment.refreshHealth() }
                .onOpenURL { environment.router.handle($0) }
                // Light is the default. Dark mode still works because every
                // colour in the app is semantic, but nothing is *designed*
                // dark, and there is no dark hero surface anywhere.
                .tint(.accentColor)
        }
    }
}

struct RootView: View {
    @Environment(AppEnvironment.self) private var environment
    @State private var showingSettings = false

    var body: some View {
        @Bindable var router = environment.router

        TabView(selection: $router.selectedTab) {
            NavigationStack(path: $router.closetPath) {
                ClosetView()
                    .navigationDestination(for: Destination.self) { destination(for: $0) }
            }
            .tabItem { Label("Closet", systemImage: "square.grid.2x2") }
            .tag(Tab.closet)

            NavigationStack(path: $router.searchPath) {
                SearchView()
                    .navigationDestination(for: Destination.self) { destination(for: $0) }
            }
            .tabItem { Label("Search", systemImage: "magnifyingglass") }
            .tag(Tab.search)

            NavigationStack(path: $router.feedPath) {
                FeedView()
                    .navigationDestination(for: Destination.self) { destination(for: $0) }
            }
            .tabItem { Label("Fits", systemImage: "sparkles") }
            .tag(Tab.feed)
        }
        .sheet(isPresented: $showingSettings) {
            SettingsView()
        }
        .environment(\.openSettings, OpenSettingsAction { showingSettings = true })
    }

    @ViewBuilder
    private func destination(for destination: Destination) -> some View {
        switch destination {
        case .outfit(let outfit):
            OutfitDetailView(outfit: outfit)
        case .closetItem(let id):
            if let item = environment.closet.items.first(where: { $0.id == id }) {
                ClosetItemDetailView(item: item)
            } else {
                EmptyStateView(title: "That item is gone",
                               explanation: "It was removed from your closet.")
            }
        }
    }
}

/// Lets any screen open Settings without every screen owning a sheet.
struct OpenSettingsAction {
    let handler: () -> Void
    func callAsFunction() { handler() }
    init(_ handler: @escaping () -> Void) { self.handler = handler }
}

private struct OpenSettingsKey: EnvironmentKey {
    static let defaultValue = OpenSettingsAction {}
}

extension EnvironmentValues {
    var openSettings: OpenSettingsAction {
        get { self[OpenSettingsKey.self] }
        set { self[OpenSettingsKey.self] = newValue }
    }
}

#Preview("Root") {
    RootView().environment(AppEnvironment.preview)
}
