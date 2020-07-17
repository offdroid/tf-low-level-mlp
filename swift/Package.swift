// swift-tools-version:5.2
import PackageDescription

let package = Package(
    name: "swift",
    dependencies: [
    ],
    targets: [
        .target(
            name: "swift",
            dependencies: []),
        .testTarget(
            name: "swiftTests",
            dependencies: ["swift"]),
    ]
)
