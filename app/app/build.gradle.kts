// SIH26168 — Dead Reckoning app module.
//
// Ports the already-validated Python pipeline (src/calibration.py,
// src/models/strapdown_ins.py, src/fusion.py) to Kotlin, running the
// exported two-model ensemble (checkpoints/model_slow.onnx and
// model_fast.onnx, bundled as assets) on-device via ONNX Runtime Mobile —
// see src/export_ensemble.py's docstring for the exact contract this app
// implements: a fixed-size calibrated-IMU window in (plus the start speed
// for the fast model), per-timestep [delta_v, delta_theta] corrections out.
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

        // Where ModelUpdater looks for manifest.json and model files: the
        // public "models" bucket in the project's Supabase Storage (see
        // scripts/publish_model.py). Override for local testing with
        // ./gradlew assembleDebug -PmodelBaseUrl=http://10.0.2.2:8765/
        val modelBaseUrl = (project.findProperty("modelBaseUrl") as String?)
            ?: "https://iekkwudnpvijroqvknwq.supabase.co/storage/v1/object/public/models/"
        buildConfigField("String", "MODEL_BASE_URL", "\"$modelBaseUrl\"")
    }

    buildFeatures {
        buildConfig = true
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

    // The exported models (checkpoints/model_slow.onnx, model_fast.onnx) are copied
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

    // On-device inference for the exported dead-reckoning network. Pinned
    // to a recent release (not the literal latest) specifically because
    // 1.18.0's bundled libonnxruntime.so/libonnxruntime4j_jni.so aren't
    // 16KB-page-size aligned (confirmed live: a real-device "Android app
    // compatibility" warning) — newer NDK toolchains build 16KB-aligned by
    // default, which a sufficiently recent release should pick up.
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")

    // Optional account/data-contribution layer (SupabaseAuthClient.kt,
    // AuthActivity.kt) — sign-up/sign-in and drive-session upload consent,
    // entirely separate from and never called by the dead-reckoning
    // pipeline itself, which stays fully offline. Plain OkHttp calling
    // Supabase's REST endpoints directly (GoTrue for auth, PostgREST for
    // the profiles/devices/drive_sessions tables — see ../../SUPABASE.md)
    // rather than the full supabase-kt SDK, which pulls in Ktor + kotlinx-
    // serialization for a Kotlin Multiplatform surface this single-platform
    // Android app doesn't need — matches this module's existing preference
    // for the smallest dependency that does the job (see the osmdroid
    // comment below).
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Real OpenStreetMap-tile map view — no API key, no Google Play
    // Services, same free-OSM-tiles philosophy as LocationReader.kt's
    // choice of plain LocationManager over FusedLocationProviderClient.
    implementation("org.osmdroid:osmdroid-android:6.1.20")

    // Plain JVM unit tests (FusionEngineTest) — no device/emulator needed,
    // runs via `./gradlew test`. FusionEngine itself has no Android
    // dependency; only BiasCorrectionModel does (Predictor exists so tests
    // can swap in a fake instead).
    testImplementation("junit:junit:4.13.2")
    // Android's own org.json is a stub on the plain JVM; this is the real
    // implementation, so ModelUpdateTest can parse manifests.
    testImplementation("org.json:json:20240303")
}

// Keeps the app's bundled models in sync with the real training artifacts
// instead of hand-copied, potentially-stale duplicates — run this (or
// just build; assembleDebug depends on it) any time
// checkpoints/model_slow.onnx or model_fast.onnx changes.
val ensembleModels = listOf("model_slow.onnx", "model_fast.onnx")
tasks.register<Copy>("syncModel") {
    from(ensembleModels.map { rootProject.file("../checkpoints/$it") })
    into(layout.projectDirectory.dir("src/main/assets"))
    doFirst {
        for (name in ensembleModels) {
            val src = rootProject.file("../checkpoints/$name")
            if (!src.exists()) {
                throw GradleException(
                    "checkpoints/$name not found at $src — " +
                    "run `python -m src.export_ensemble` in the project root first."
                )
            }
        }
    }
}
tasks.named("preBuild") { dependsOn("syncModel") }
