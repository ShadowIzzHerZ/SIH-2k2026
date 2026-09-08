// SIH26168 — Dead Reckoning app module.
//
// Ports the already-validated Python pipeline (src/calibration.py,
// src/models/strapdown_ins.py, src/fusion.py) to Kotlin, running the
// exported ONNX model (checkpoints/dead_reckoning_model.onnx, bundled as
// an asset) on-device via ONNX Runtime Mobile — see src/export_onnx.py's
// docstring for the exact contract this app implements: a fixed-size
// calibrated-IMU window in, per-timestep [delta_v, delta_theta]
// corrections out.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.sih26168.deadreckoning"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.sih26168.deadreckoning"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // Every realistic demo/judging device is arm64 — bundling
        // onnxruntime-android's armeabi-v7a/x86/x86_64 .so's too roughly
        // quadruples APK size for ABIs nothing here will ever run on.
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    // The exported model (checkpoints/dead_reckoning_model.onnx) is copied
    // into src/main/assets by the sync-model Gradle task below, not
    // committed twice — see that task's doc comment.
    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    // On-device inference for the exported dead-reckoning network.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // Real OpenStreetMap-tile map view — no API key, no Google Play
    // Services, same free-OSM-tiles philosophy as LocationReader.kt's
    // choice of plain LocationManager over FusedLocationProviderClient.
    implementation("org.osmdroid:osmdroid-android:6.1.20")
}

// Keeps the app's bundled model in sync with the real training artifact
// instead of a hand-copied, potentially-stale duplicate — run this (or
// just build; assembleDebug depends on it) any time
// checkpoints/dead_reckoning_model.onnx changes.
tasks.register<Copy>("syncModel") {
    from(rootProject.file("../checkpoints/dead_reckoning_model.onnx"))
    into(layout.projectDirectory.dir("src/main/assets"))
    doFirst {
        val src = rootProject.file("../checkpoints/dead_reckoning_model.onnx")
        if (!src.exists()) {
            throw GradleException(
                "checkpoints/dead_reckoning_model.onnx not found at $src — " +
                "run `python -m src.export_onnx` in the project root first."
            )
        }
    }
}
tasks.named("preBuild") { dependsOn("syncModel") }
