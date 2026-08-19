import SwiftUI

/// Loading, described honestly over time.
///
/// This backend cold-starts in about twenty seconds. A bare spinner for
/// twenty seconds reads as "broken", and users kill the app. So the wait
/// explains itself: it stays quiet for the first few seconds, then says what
/// is actually happening and roughly how long it takes. No fake progress bar
/// — nothing here knows the real progress, and a bar that lies is worse than
/// a phrase that doesn't.
@MainActor
@Observable
final class LoadingClock {
    private(set) var elapsed: TimeInterval = 0
    private var task: Task<Void, Never>?

    /// Below this, the wait is normal and needs no explanation.
    private let explainAfter: TimeInterval = 4

    var message: String? {
        guard elapsed >= explainAfter else { return nil }
        return "The server sleeps when it’s idle. The first request after that takes about 20 seconds."
    }

    func start() {
        stop()
        elapsed = 0
        task = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                guard let self else { return }
                self.elapsed += 0.5
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    // No `deinit` cancellation: the property is main-actor isolated and
    // `deinit` is not, so touching it there doesn't compile. It isn't needed
    // either — the loop captures `self` weakly and returns the moment it is
    // gone, so the task ends on its own within half a second of dealloc.
}

/// The loading view itself. Used everywhere something is in flight.
struct LoadingView: View {
    let clock: LoadingClock
    var label: String = "Searching"

    var body: some View {
        VStack(spacing: Theme.Space.snug) {
            ProgressView()
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if let message = clock.message {
                Text(message)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, Theme.Space.section)
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, Theme.Space.section)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(clock.message.map { "\(label). \($0)" } ?? label)
    }
}
