# SIH portal submission draft

Draft text for the SIH idea-submission form (Idea Title / Idea Description /
Abstract-Summary fields). Grounded in the real, measured numbers already
written up in [docs/understanding.md](understanding.md) and the problem
statement quoted in [JUDGES.md](../JUDGES.md). Copy straight into the form;
edit if anything reads wrong to you first.

## Idea Title

Zen: Phone-Sensor Dead Reckoning for GPS-Denied Vehicle Navigation

(68 characters, well under the 100-character limit.)

## Idea Description

A Kerala trekker who spent four days lost in a forest in Karnataka before
finding her way back out is one of the extreme situations we are trying
to solve. Most people never end up that far from help, but the same gap
that stranded her, a signal disappearing with nothing left to lean on,
shows up constantly on ordinary Indian roads too.

Picture an ambulance driver taking the underground ramp into a hospital
and watching the blue dot on the dashboard freeze mid-turn. Or a delivery
rider threading through a market lane boxed in by four-story buildings,
the app quietly losing track of which alley he is actually in. Most
vehicles on Indian roads have no dedicated inertial navigation hardware,
so the moment GPS drops in a tunnel, an underground parking structure, a
dense street, or under tree cover, the navigation app simply stops
updating. There is no fallback.

We built a system that keeps estimating vehicle position through that gap
using only sensors already inside the phone: the accelerometer and
gyroscope. Raw physics-only integration of those sensors fails quickly,
because MEMS sensors carry a small bias that wanders and compounds every
second. So instead of trusting the raw integration, we trained a neural
network to predict and correct that bias in real time, using two public
driving datasets (comma2k19 and IO-VNBD) plus our own recordings, collected
by walking and driving around Jalandhar, Punjab with a real phone.

The pipeline runs in four stages. First, a calibration step works out how
the phone is actually held or mounted relative to the vehicle, since
nobody holds a phone perfectly level or perfectly aligned with the
direction of travel. Second, a fusion engine switches between three
states: normal GPS tracking, INS-only tracking during a blackout, and a
short blend when GPS returns, so the displayed position never jumps.
Third, the trained model corrects the IMU's drift in short chunks while
GPS is unavailable. Fourth, the corrected trajectory gets snapped onto the
real road network using OpenStreetMap data and a map matcher, because a
car cannot drive through a building or slide sideways off a road; using
that constraint to clean up a drifted position estimate cut error by 62%
in our tests.

On a 30-second continuous GPS blackout using real recorded highway
driving, the system held drift to 2.9%, under the problem statement's 10%
target. On the harder case of slow, stop-and-go urban driving, the current
model is weaker, closer to 60% drift, and we are not hiding that number.
The gap comes from how little the training data represents low-speed
urban stop-and-go compared to steady highway cruising, and it is the
clearest next place to put more real-world recordings.

Everything runs on the phone itself. The trained model is exported to
ONNX and executed on-device through ONNX Runtime Mobile, so the core
navigation has no dependency on a live server connection. A server
connection is only used optionally, to sync recorded drive data afterward
through an opt-in account layer. The Android app shows a real
OpenStreetMap map, a live position marker, and a manual blackout toggle,
so the failure mode can be demonstrated on stage without actually driving
into a tunnel.

When GPS comes back, whether it is the ambulance clearing the ramp or the
rider popping back out into the open street, the position on screen does
not jump to catch up. It settles in where it should have been the whole
time.

(About 470 words, well under the 50,000-character limit.)

## Abstract/Summary

Indian vehicles generally lack dedicated inertial navigation hardware, so
a lost GPS signal in a tunnel, underground parking, a dense urban canyon,
or a forest leaves standard navigation apps with no fallback. This project
addresses SIH26168 by using a phone's built-in accelerometer and gyroscope
to keep estimating vehicle position through a GPS blackout. Raw sensor
integration drifts too fast to be useful on its own, so we trained a
neural network on real driving data (comma2k19, IO-VNBD, and our own
recordings from Jalandhar, Punjab) to correct that drift in real time. The
system combines phone-to-vehicle calibration, a GNSS/INS fusion state
machine with smooth transitions between GPS tracking, blackout, and
reconnect, and map-matching against OpenStreetMap road data to keep the
estimated position physically plausible. On a 30-second continuous
blackout using real highway driving data, measured drift was 2.9%, under
the problem statement's 10% target; on slow urban stop-and-go driving,
drift is currently closer to 60%, a limitation we are addressing with more
real-world low-speed recordings. The full pipeline runs on-device on
Android through ONNX Runtime Mobile, with no server dependency for core
navigation.

(About 190 words, well under the 10,000-character limit.)

## Fields not drafted here

- **Idea Template (PDF upload):** needs the official SIH template filled
  in and exported as a PDF; not something to draft as plain text.
- **Technology Bucket:** pick whatever the dropdown actually offers that's
  closest to AI/ML or Smart Vehicles/Transportation; the exact option list
  isn't visible from here, so check the live dropdown rather than trust a
  guess.
- **YouTube Link (optional):** leave blank until there's a real demo video
  to link.
