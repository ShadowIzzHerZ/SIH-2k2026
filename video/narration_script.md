# Narration script

One paragraph per on-screen slide, in video order. Voiced with the Kokoro neural TTS model (voice af_heart). Re-record in your own voice if you prefer.

**1. title**
We're Team Zen, and this is our Smart India Hackathon 2026 entry. Problem statement SIH26168, from ISRO: AI and machine learning based dead reckoning for seamless navigation.

**2. prob_trek**
A trekker from Kerala once spent four days lost in a Karnataka forest, with no signal to guide her out.

**3. prob_roads**
The same gap hits ordinary roads. An ambulance drops into an underground ramp and the blue dot freezes. A delivery rider enters a lane boxed in by tall buildings and the app loses the alley.

**4. prob_nofallback**
Most vehicles in India have no inertial backup, so when GPS drops, navigation just stops.

**5. idea_drift**
Our idea: use the accelerometer and gyroscope every phone already has. Adding up their readings drifts within seconds, because tiny sensor errors compound.

**6. idea_hybrid**
So we don't ask a network to guess position. A small network learns only how wrong the sensors are, exact physics does the rest, and the whole chain trains on the drift percentage the problem statement grades.

**7. arch_full**
Here's the whole system. Offline, we train in PyTorch and export an ONNX model into the Android app, which runs everything on the phone.

**8. arch_train**
We train on the public IO-VNBD and comma2k19 datasets, plus our own drives from Jalandhar.

**9. arch_app**
On the phone, ten times a second, sensor readings go through calibration and into the fusion engine.

**10. arch_states**
While GPS is good, we trust it. When it drops, blackout mode runs the model in five second chunks. When GPS returns, we blend back over two seconds, so the marker never jumps.

**11. arch_matcher**
A map matcher then snaps the estimate onto the real road graph, because a car can't drive through a building.

**12. demo2**
On a real recorded highway drive, we cut GPS for thirty seconds. The red trail is the model alone, and it ends with 2.9 percent drift, under the 10 percent target. When GPS returns, the jump is zero metres.

**13. bug_stuck**
Now the honest part. Our urban results were stuck near 62 percent drift, and five different fixes didn't move them.

**14. bug_labels**
So we checked the data. In IO-VNBD the speed column says kilometres per hour but is really metres per second, so every speed label was 3.6 times too small. The GPS labels were also stair steps.

**15. bug_chart**
After fixing both and retraining, urban drift on the same test set fell from 77 percent mean to 19, with a median of 11.2.

**16. results_table**
On highway driving the median is 5.8 percent. Urban stop and go is still the hardest case. Next, we ship the retrained model in the app and re-validate it end to end.

**17. feasibility**
It's feasible today: standard phone sensors, fully on-device at ten hertz, testable with or without GPS, and no new infrastructure.

**18. impact**
Navigation keeps working in tunnels, parking and dense streets with no sudden jump. That helps ambulances, fleets and delivery riders, and it scales to robots and drones.

**19. refs**
We build on published research, with references on screen. Thank you from Team Zen.

