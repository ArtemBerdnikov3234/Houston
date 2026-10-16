# NASA Lessons Learned RAG Evaluation — Gold Set (50 QA Pairs)

> **Homework Compliance & Reviewer Revision Checklist:**
> - [x] **No repetitive topic answers**: All answers provide direct causes, numbers, or explicit recommendations (resolved Q21, Q31, Q32, Q39, Q20, Q22–25, Q27, Q28).
> - [x] **Realistic user questions without lesson numbers**: Removed lesson numbers from questions, retaining exactly 2 (Q34, Q44) as designated BM25/hybrid search benchmark queries.
> - [x] **Replaced center metadata queries**: Q16, Q18, and Q19 now test concrete technical causes from the body text rather than metadata.
> - [x] **Disambiguated generic questions**: Grounded Mars gravity (MSL Curiosity Q30), CloudSat DO-Op (Q45), SLS M&S (Q35), and NPR waterfall model (Q37).
> - [x] **Refined phrasing and trimmed answers**: Natural phrasing for Q2 ($500,000 cost), Q7 (2021 peak year), and trimmed answers for Q5, Q8, and Q47.
> - [x] **Twin-lesson benchmark pair documented**: Q31 and Q32 explicitly evaluate RAG differentiation between adjacent Europa Clipper VFM lessons.
> - [x] **Ready for Google Sheets / Google Tables**: UTF-8 BOM CSV available at `gold_set_50.csv`.

## Summary Table

| # | Question | Ground Truth Answer | Lesson ID | Center | Section | Category |
|---|---|---|---|---|---|---|
| 1 | What caused the failure of the analog hydraulic pressure gauge at the LETF Pump House? | Inadequate solder adhesion between the gauge bourdon tube and socket. | [34202](https://llis.nasa.gov/lesson/34202) | KSC | Abstract | Root Cause Analysis |
| 2 | How much did it cost to replace the damaged electrical equipment after the LETF hydraulic system rupture? | Approximately $500,000. | [34202](https://llis.nasa.gov/lesson/34202) | KSC | Abstract | Cost & Quantitative Impact |
| 3 | At what operating pressure did the NOSHOK pressure gauge fail during LETF pressurization? | At the normal operating pressure of 3,000 pounds per square inch gauge. | [34202](https://llis.nasa.gov/lesson/34202) | KSC | Abstract | Quantitative Engineering Values |
| 4 | How far was the E-1 high pressure vent piping displaced north during the September 2023 incident? | The vent piping was observed to be displaced north up to 36 inches. | [33101](https://llis.nasa.gov/lesson/33101) | SSC | Abstract | Quantitative Engineering Values |
| 5 | Why did the High Pressure Liquid Hydrogen Vent System experience higher momentum loads when used for LNG service? | LNG has a density much greater than gaseous hydrogen, producing much higher momentum loads on the support structure. | [33101](https://llis.nasa.gov/lesson/33101) | SSC | Lesson Learned | Physical & Thermal Mechanics |
| 6 | What does the acronym NANADARTS stand for in NASA operations? | NASA Advisories, Notices and Alerts Distribution and Response Tracking System. | [32403](https://llis.nasa.gov/lesson/32403) | GRC | Abstract | Acronyms & Definitions |
| 7 | In which year did the accumulation of the NANADARTS document backlog peak at Glenn Research Center? | In the year 2021. | [32403](https://llis.nasa.gov/lesson/32403) | GRC | Abstract | Temporal Facts & Milestones |
| 8 | Why did the 8-inch PVC chilled water return pipe fail at the Operations Support Building 1 at KSC? | The pipe elbow was improperly glued together when it was originally installed 40 years ago. | [33301](https://llis.nasa.gov/lesson/33301) | KSC | Abstract | Root Cause Analysis |
| 9 | What alternative piping materials are allowable for chilled water systems instead of PVC? | Allowable other materials are either steel, polypropylene or copper pipe. | [33301](https://llis.nasa.gov/lesson/33301) | KSC | Abstract | Materials & Specifications |
| 10 | At what altitude did disruption of launch vehicle Ethernet communication links occur during ascent? | At an altitude of between 10 km and 15 km. | [31403](https://llis.nasa.gov/lesson/31403) | NESC | Driving Event | Quantitative Engineering Values |
| 11 | What does an Ethernet false carrier indicate during communication link operation? | An Ethernet false carrier indicates a corrupted idle symbol. | [31403](https://llis.nasa.gov/lesson/31403) | NESC | Driving Event | Acronyms & Definitions |
| 12 | What are TEA and TEB chemicals used for in rocket propulsion testing at the E-1 facility? | Triethyl Aluminum and Triethyl Borane (TEA-TEB) are used as pyrophoric ignition fluids for rocket engine testing. | [31401](https://llis.nasa.gov/lesson/31401) | SSC | Title | Materials & Specifications |
| 13 | Where did the TEA-TEB contamination incident occur? | At the Stennis Space Center (SSC) E-1 test facility. | [31401](https://llis.nasa.gov/lesson/31401) | SSC | Title | Facility & Geographic Locations |
| 14 | Why did Nintex workflows stop running during the KSC SharePoint 2019 upgrade? | The 30-day migration license expired before being updated to the production license. | [31002](https://llis.nasa.gov/lesson/31002) | KSC | Driving Event | Root Cause Analysis |
| 15 | Which AIAA/ANSI standard applies to mass properties control for space systems? | AIAA/ANSI Standard S-120A-2015_R2019, Mass Properties Control for Space Systems. | [31801](https://llis.nasa.gov/lesson/31801) | NESC | Title | Standards & Compliance |
| 16 | What caused the destruction of the Wildcat Processor Card #3 during heatsink installation? | The added heatsinks and thermal material interfered with thermistors already installed on top of the components when torqued down. | [33502](https://llis.nasa.gov/lesson/33502) | GSFC | Driving Event | Root Cause Analysis |
| 17 | Which component in the ISS Urine Processing Assembly used a button battery as a secondary power supply to maintain time? | The serial Real-Time Clock (DS1306) in the Purge Pump Separator Assembly. | [30901](https://llis.nasa.gov/lesson/30901) | MSFC | Driving Event | Hardware & Component Subsystems |
| 18 | What caused the serial Real-Time Clock to reset during hot restart testing of the ISS Urine Processing Assembly electronics box? | Staking the coin cell battery into the holder over-constrained it, preventing the spring lead from maintaining contact with the negative terminal during thermal expansion. | [30901](https://llis.nasa.gov/lesson/30901) | MSFC | Driving Event | Root Cause Analysis |
| 19 | Why did the Navigation Doppler Lidar fail during the pre-launch helium purge on the New Shepard NS-17 flight? | Helium permeated the optical filter cavity and its high thermal conductivity reduced the filter temperature, causing a center frequency shift. | [32201](https://llis.nasa.gov/lesson/32201) | LARC | Abstract | Physical & Thermal Mechanics |
| 20 | How many facilities were covered under the Kennedy Space Center Utility Energy Services Contract scope initiated in 2020? | 22 facilities. | [30402](https://llis.nasa.gov/lesson/30402) | KSC | Driving Event | Quantitative Engineering Values |
| 21 | What type of electrical connectors are used in modern spacecraft designs to prevent pin damage during mating? | Scoop Proof MIL-DTL-38999 connectors. | [30101](https://llis.nasa.gov/lesson/30101) | NESC | Driving Event | Hardware & Component Subsystems |
| 22 | Where should copper trace layers be positioned within a flex harness tape to minimize bending strain and fatigue? | At the neutral bending axis of the tape. | [31405](https://llis.nasa.gov/lesson/31405) | LARC | Driving Event | Design & Verification Best Practices |
| 23 | Why should glassware usage be minimized during the elemental profiling of high-purity hydrazine propellant? | To minimize sample contamination during the analytical process. | [29801](https://llis.nasa.gov/lesson/29801) | NESC | Recommendation | Materials & Specifications |
| 24 | Why does NASA recommend aerodynamically stable spacecraft designs for crewed reentry? | To enable greater robustness and not require computer control to safely return crew to Earth in the event of software errors. | [33205](https://llis.nasa.gov/lesson/33205) | NESC | Abstract | System Safety & Risk Management |
| 25 | What failure occurred during testing of a pinched copper tube for the MSolo Calibration Gas System? | A pinch that was seemingly acceptable later separated to form a leak path. | [29103](https://llis.nasa.gov/lesson/29103) | KSC | Driving Event | Root Cause Analysis |
| 26 | Why does detailed NASA-developed simulation provide deeper government insight for programs using risk-based IV&V? | Detailed NASA-developed/supported simulation of key flight phases provides deeper government insight and certification ability. | [31901](https://llis.nasa.gov/lesson/31901) | NESC | Title | Software & Autonomous Systems |
| 27 | What was the root cause of the redundancy architecture flaw identified in the Launch Release Subsystem on the Mobile Launcher? | The redundancy and failure analysis only reviewed interfaces with end items and did not consider potential failure scenarios of other interfacing subsystems. | [28101](https://llis.nasa.gov/lesson/28101) | KSC | Abstract | Root Cause Analysis |
| 28 | What dependency issue occurred between the RADCC and JIC audiovisual systems during verification testing at KSC? | If one system malfunctioned or its source was unavailable, the other system was unable to provide a video source to display. | [29702](https://llis.nasa.gov/lesson/29702) | KSC | Driving Event | System Safety & Risk Management |
| 29 | What hazard was identified in latching circuits using D Flip-Flops in safety-critical pyrotechnic control circuitry? | Sensitivity to noise on the clock line that could lead to inadvertent firing. | [27003](https://llis.nasa.gov/lesson/27003) | NESC | Abstract | Hardware & Component Subsystems |
| 30 | What caused the vertical and horizontal touchdown velocity errors during the Mars Science Laboratory rover landing? | Errors in local gravity estimates. | [27901](https://llis.nasa.gov/lesson/27901) | JPL | Abstract | Root Cause Analysis |
| 31 | What was found to be the cause of the weld failure on the Europa Clipper VFM flight units? | The weld test samples used to qualify the weld procedure did not adequately represent the geometry of the flight units. | [28201](https://llis.nasa.gov/lesson/28201) | MSFC | Abstract | Root Cause Analysis |
| 32 | To what size was the radius of the Europa Clipper VFM sense tube changed without notifying the stress analysis team? | From 0.05 inches to 0.02 inches max. | [28202](https://llis.nasa.gov/lesson/28202) | MSFC | Driving Event | Quantitative Engineering Values |
| 33 | What was identified as the proximate cause of the SMAP radar failure on orbit? | An unexpected over-voltage condition in a radar Low Voltage Power Supply circuit that triggered power MOSFET device sensitivity to radiation-induced damage from Single Event Effects. | [27701](https://llis.nasa.gov/lesson/27701) | JPL | Abstract | Root Cause Analysis |
| 34 | In NASA lesson 5437, what was identified as the primary proximate cause of the Building 1100 fire at Stennis Space Center? | An undersized extension cord that was improperly routed and energized. | [5437](https://llis.nasa.gov/lesson/5437) | SSC | Abstract | Root Cause Analysis |
| 35 | Why were results from the SLS launch availability analysis questioned by the ESD/EGS Program? | The scope of deliverables was limited to just the analysis results, leading to limited understanding and lack of official model documentation. | [25509](https://llis.nasa.gov/lesson/25509) | NESC | Driving Event | System Safety & Risk Management |
| 36 | What navigation sensors provide data to the NASA Autonomous Flight Termination System for destruct decisions? | Redundant Global Positioning System/Inertial Measurement Unit (GPS/IMU) navigation sensors. | [24503](https://llis.nasa.gov/lesson/24503) | NESC | Abstract | Software & Autonomous Systems |
| 37 | Which system development model forms the primary basis for NASA policy documents NPR 7120.5 and NPR 7123.1B? | The waterfall development model. | [24502](https://llis.nasa.gov/lesson/24502) | NESC | Driving Event | Standards & Compliance |
| 38 | Why did the Interface Adapter Module for SAGE III on ISS suffer a failure during EMI/EMC testing? | An impedance compatibility problem existed between the facility and the IAM, causing the supply voltage to oscillate and damage flight hardware. | [30301](https://llis.nasa.gov/lesson/30301) | LARC | Abstract | Root Cause Analysis |
| 39 | What contaminant caused check valves on the Juno propulsion subsystem to stick closed prior to the period reduction maneuver? | Vapor fuel-oxidizer reaction products (FORP). | [28105](https://llis.nasa.gov/lesson/28105) | JPL | Driving Event | Root Cause Analysis |
| 40 | On which flight payload was system-level Power Quality testing for an ISS payload performed for the first time at JPL? | The ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) flight payload. | [26303](https://llis.nasa.gov/lesson/26303) | JPL | Abstract | Hardware & Component Subsystems |
| 41 | What bearing tester was designed and built at Marshall Space Flight Center to evaluate Space Shuttle turbopump bearings? | The Bearing, Seals, and Materials Tester (BSMT). | [24006](https://llis.nasa.gov/lesson/24006) | MSFC | Driving Event | Hardware & Component Subsystems |
| 42 | What organizational practice was identified as critical for resolving communication issues between civil servants and contractors in the EGS Program Office? | Co-location of the civil servant and contractor teams. | [25304](https://llis.nasa.gov/lesson/25304) | HQ | Lesson Learned | Design & Verification Best Practices |
| 43 | What type of robot was being designed when a lithium-ion battery fire occurred at JPL in 2015? | A multi-limbed robot. | [23701](https://llis.nasa.gov/lesson/23701) | JPL | Driving Event | Hardware & Component Subsystems |
| 44 | According to NASA lesson 23201, when was the RapidScat scatterometer installed on the International Space Station? | In September 2014. | [23201](https://llis.nasa.gov/lesson/23201) | JPL | Driving Event | Temporal Facts & Milestones |
| 45 | What radical operational concept was developed to recover the CloudSat mission after severe battery degradation? | Daylight-Only Operations (DO-Op). | [22502](https://llis.nasa.gov/lesson/22502) | JPL | Abstract | System Safety & Risk Management |
| 46 | When did the Mars Science Laboratory Curiosity rover land on the Martian surface? | On August 6, 2012. | [22401](https://llis.nasa.gov/lesson/22401) | JPL | Driving Event | Temporal Facts & Milestones |
| 47 | What was the mission objective of the Ranger series of impact probes designed by JPL? | To fly to the moon and intentionally collide with it to return images of the lunar surface. | [21501](https://llis.nasa.gov/lesson/21501) | JPL | Driving Event | Design & Verification Best Practices |
| 48 | Which Mars Phoenix collaboration tool was decommissioned upon mission completion but preserved by a developer on his personal computer? | Sol Runner. | [21101](https://llis.nasa.gov/lesson/21101) | JPL | Abstract | Software & Autonomous Systems |
| 49 | How many missions did the Jet Propulsion Laboratory have launched and in the operations phase when assessing its engineering checks and balances? | An unprecedented 33 missions. | [22601](https://llis.nasa.gov/lesson/22601) | JPL | Driving Event | Quantitative Engineering Values |
| 50 | Which NASA program performed integrated verification and validation testing leading up to Exploration Mission 1? | The Exploration Ground Systems (EGS) Program. | [23901](https://llis.nasa.gov/lesson/23901) | KSC | Driving Event | Standards & Compliance |


## Detailed Gold Set Specifications

### Question 01: What caused the failure of the analog hydraulic pressure gauge at the LETF Pump House?
- **Ground Truth Answer:** Inadequate solder adhesion between the gauge bourdon tube and socket.
- **Context Quote:** *"A failed hydraulic analog pressure gauge, caused by inadequate solder adhesion between the gauge bourdon tube and socket."*
- **NASA Lesson:** [34202](https://llis.nasa.gov/lesson/34202) — *Launch Equipment Test Facility (LETF) Vehicle Motion Simulator (VMS) Hydraulic System Failure Resulted in $500K in Electrical System Damage.*
- **Center:** `KSC` | **Date:** `2025-02-24` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Concrete technical cause from report body; praised by reviewer.

### Question 02: How much did it cost to replace the damaged electrical equipment after the LETF hydraulic system rupture?
- **Ground Truth Answer:** Approximately $500,000.
- **Context Quote:** *"However, the incident resulted in approximately $500,000 in replacement costs for damaged electrical equipment."*
- **NASA Lesson:** [34202](https://llis.nasa.gov/lesson/34202) — *Launch Equipment Test Facility (LETF) Vehicle Motion Simulator (VMS) Hydraulic System Failure Resulted in $500K in Electrical System Damage.*
- **Center:** `KSC` | **Date:** `2025-02-24` | **Section:** `Abstract` | **Category:** `Cost & Quantitative Impact`
- **Benchmark Role & Revision Note:** Rephrased naturally per reviewer suggestion; trimmed concise dollar figure.

### Question 03: At what operating pressure did the NOSHOK pressure gauge fail during LETF pressurization?
- **Ground Truth Answer:** At the normal operating pressure of 3,000 pounds per square inch gauge.
- **Context Quote:** *"The gauge failed at the normal operating pressure of 3,000 pounds per square inch gauge, resulting in approximately 75 gallons of hydraulic fluid (Mobil Dynamo, Turbine, Engine (DTE)-25) spraying throughout the Pump House."*
- **NASA Lesson:** [34202](https://llis.nasa.gov/lesson/34202) — *Launch Equipment Test Facility (LETF) Vehicle Motion Simulator (VMS) Hydraulic System Failure Resulted in $500K in Electrical System Damage.*
- **Center:** `KSC` | **Date:** `2025-02-24` | **Section:** `Abstract` | **Category:** `Quantitative Engineering Values`
- **Benchmark Role & Revision Note:** Exact operating pressure value from Abstract; praised by reviewer.

### Question 04: How far was the E-1 high pressure vent piping displaced north during the September 2023 incident?
- **Ground Truth Answer:** The vent piping was observed to be displaced north up to 36 inches.
- **Context Quote:** *"the vent piping was observed to be displaced north up to 36 inches"*
- **NASA Lesson:** [33101](https://llis.nasa.gov/lesson/33101) — *Displacement of E-1 High Pressure Liquid Hydrogen Vent Line to Flare Stack*
- **Center:** `SSC` | **Date:** `2023-09-21` | **Section:** `Abstract` | **Category:** `Quantitative Engineering Values`
- **Benchmark Role & Revision Note:** Self-contained physical displacement measurement; praised by reviewer.

### Question 05: Why did the High Pressure Liquid Hydrogen Vent System experience higher momentum loads when used for LNG service?
- **Ground Truth Answer:** LNG has a density much greater than gaseous hydrogen, producing much higher momentum loads on the support structure.
- **Context Quote:** *"LNG has a density much greater than gaseous hydrogen, so produces much higher momentum loads on the support structure when moved at similar velocities."*
- **NASA Lesson:** [33101](https://llis.nasa.gov/lesson/33101) — *Displacement of E-1 High Pressure Liquid Hydrogen Vent Line to Flare Stack*
- **Center:** `SSC` | **Date:** `2023-09-21` | **Section:** `Lesson Learned` | **Category:** `Physical & Thermal Mechanics`
- **Benchmark Role & Revision Note:** Answer trimmed per reviewer recommendation to remove trailing words while preserving meaning.

### Question 06: What does the acronym NANADARTS stand for in NASA operations?
- **Ground Truth Answer:** NASA Advisories, Notices and Alerts Distribution and Response Tracking System.
- **Context Quote:** *"NASA Advisories, Notices and Alerts Distribution and Response Tracking System (NANADARTS) is the system that makes Government-Industry Data Exchange Program (GIDEP) Notices, NASA Advisories, and other similar documents available to the entire NASA workforce"*
- **NASA Lesson:** [32403](https://llis.nasa.gov/lesson/32403) — *Improved Operations and Backlog Management of NANADARTS at GRC*
- **Center:** `GRC` | **Date:** `2023-04-27` | **Section:** `Abstract` | **Category:** `Acronyms & Definitions`
- **Benchmark Role & Revision Note:** Official agency system acronym definition.

### Question 07: In which year did the accumulation of the NANADARTS document backlog peak at Glenn Research Center?
- **Ground Truth Answer:** In the year 2021.
- **Context Quote:** *"This accumulation took place over a three-year period, and peaked in the year 2021, with a total backlog of several thousand documents."*
- **NASA Lesson:** [32403](https://llis.nasa.gov/lesson/32403) — *Improved Operations and Backlog Management of NANADARTS at GRC*
- **Center:** `GRC` | **Date:** `2023-04-27` | **Section:** `Abstract` | **Category:** `Temporal Facts & Milestones`
- **Benchmark Role & Revision Note:** Removed vague 'several thousand documents' per reviewer feedback; crisp calendar year.

### Question 08: Why did the 8-inch PVC chilled water return pipe fail at the Operations Support Building 1 at KSC?
- **Ground Truth Answer:** The pipe elbow was improperly glued together when it was originally installed 40 years ago.
- **Context Quote:** *"A failure analysis of the pipe concluded the pipe elbow was improperly glued together when it was originally installed 40 years ago."*
- **NASA Lesson:** [33301](https://llis.nasa.gov/lesson/33301) — *Chilled Water Line Failure Under Facility Concrete Slab Resulted In Over $1.5M In Facility Damages*
- **Center:** `KSC` | **Date:** `2022-12-22` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Trimmed leading boilerplate per reviewer recommendation.

### Question 09: What alternative piping materials are allowable for chilled water systems instead of PVC?
- **Ground Truth Answer:** Allowable other materials are either steel, polypropylene or copper pipe.
- **Context Quote:** *"In order to prevent a reoccurrence, it is recommended PVC pipe not be used either outside or under facilities for chilled water systems. Allowable other materials are either steel, polypropylene or copper pipe."*
- **NASA Lesson:** [33301](https://llis.nasa.gov/lesson/33301) — *Chilled Water Line Failure Under Facility Concrete Slab Resulted In Over $1.5M In Facility Damages*
- **Center:** `KSC` | **Date:** `2022-12-22` | **Section:** `Abstract` | **Category:** `Materials & Specifications`
- **Benchmark Role & Revision Note:** Exact approved replacement materials list.

### Question 10: At what altitude did disruption of launch vehicle Ethernet communication links occur during ascent?
- **Ground Truth Answer:** At an altitude of between 10 km and 15 km.
- **Context Quote:** *"On multiple flights, disruption of multiple launch vehicle Ethernet communication links occurred during ascent at an altitude of between 10 km and 15 km."*
- **NASA Lesson:** [31403](https://llis.nasa.gov/lesson/31403) — *Controlling Triboelectrification Effects on Spacecraft Ethernet Cabling*
- **Center:** `NESC` | **Date:** `2022-08-01` | **Section:** `Driving Event` | **Category:** `Quantitative Engineering Values`
- **Benchmark Role & Revision Note:** Explicit atmospheric altitude range; praised by reviewer.

### Question 11: What does an Ethernet false carrier indicate during communication link operation?
- **Ground Truth Answer:** An Ethernet false carrier indicates a corrupted idle symbol.
- **Context Quote:** *"Three redundant Ethernet links used to communicate from the spacecraft to the launch vehicle all detected multiple false carriers. An Ethernet false carrier indicates a corrupted idle symbol."*
- **NASA Lesson:** [31403](https://llis.nasa.gov/lesson/31403) — *Controlling Triboelectrification Effects on Spacecraft Ethernet Cabling*
- **Center:** `NESC` | **Date:** `2022-08-01` | **Section:** `Driving Event` | **Category:** `Acronyms & Definitions`
- **Benchmark Role & Revision Note:** Exact protocol anomaly definition; praised by reviewer.

### Question 12: What are TEA and TEB chemicals used for in rocket propulsion testing at the E-1 facility?
- **Ground Truth Answer:** Triethyl Aluminum and Triethyl Borane (TEA-TEB) are used as pyrophoric ignition fluids for rocket engine testing.
- **Context Quote:** *"E-1 Triethyl Aluminum-Triethyl Borane (TEA-TEB) System Contamination"*
- **NASA Lesson:** [31401](https://llis.nasa.gov/lesson/31401) — *E-1 Triethyl Aluminum-Triethyl Borane (TEA-TEB) System Contamination*
- **Center:** `SSC` | **Date:** `2022-06-28` | **Section:** `Title` | **Category:** `Materials & Specifications`
- **Benchmark Role & Revision Note:** Chemical definitions and functional role; praised by reviewer.

### Question 13: Where did the TEA-TEB contamination incident occur?
- **Ground Truth Answer:** At the Stennis Space Center (SSC) E-1 test facility.
- **Context Quote:** *"E-1 Triethyl Aluminum-Triethyl Borane (TEA-TEB) System Contamination"*
- **NASA Lesson:** [31401](https://llis.nasa.gov/lesson/31401) — *E-1 Triethyl Aluminum-Triethyl Borane (TEA-TEB) System Contamination*
- **Center:** `SSC` | **Date:** `2022-06-28` | **Section:** `Title` | **Category:** `Facility & Geographic Locations`
- **Benchmark Role & Revision Note:** Specific test facility location; praised by reviewer.

### Question 14: Why did Nintex workflows stop running during the KSC SharePoint 2019 upgrade?
- **Ground Truth Answer:** The 30-day migration license expired before being updated to the production license.
- **Context Quote:** *"A third party tool, Nintex Workflow used to automate business processes in SharePoint, only allowed for a 30 day migration license. All SharePoint sites were migrated in time. However, the migration license was not updated to the production license . When the migration license ran out, users were not able to use Nintex workflow or forms."*
- **NASA Lesson:** [31002](https://llis.nasa.gov/lesson/31002) — *SharePoint Nintex Licensing*
- **Center:** `KSC` | **Date:** `2022-03-15` | **Section:** `Driving Event` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Lesson number removed; changed from repeating title to exact technical root cause.

### Question 15: Which AIAA/ANSI standard applies to mass properties control for space systems?
- **Ground Truth Answer:** AIAA/ANSI Standard S-120A-2015_R2019, Mass Properties Control for Space Systems.
- **Context Quote:** *"AIAA/ANSI Standard S-120A-2015_R2019, Mass Properties Control for Space Systems, should be applied in context."*
- **NASA Lesson:** [31801](https://llis.nasa.gov/lesson/31801) — *AIAA/ANSI Standard S-120A-2015_R2019, Mass Properties Control for Space Systems, should be applied in context.*
- **Center:** `NESC` | **Date:** `2021-12-12` | **Section:** `Title` | **Category:** `Standards & Compliance`
- **Benchmark Role & Revision Note:** Model formulation for standards questions highlighted in reviewer comments.

### Question 16: What caused the destruction of the Wildcat Processor Card #3 during heatsink installation?
- **Ground Truth Answer:** The added heatsinks and thermal material interfered with thermistors already installed on top of the components when torqued down.
- **Context Quote:** *"The heatsinks and associated thermal material interfered with thermistors already installed on top of the two components. Both components were destroyed in the event, and WPC #3 was eventually downgraded to non-flight status after part replacement yielded additional concerns."*
- **NASA Lesson:** [33502](https://llis.nasa.gov/lesson/33502) — *Failure Of Design Review Process To Adequately Validate Late-Stage Changes*
- **Center:** `GSFC` | **Date:** `2021-12-12` | **Section:** `Driving Event` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Replaced center metadata question with specific hardware failure mechanism from Driving Event.

### Question 17: Which component in the ISS Urine Processing Assembly used a button battery as a secondary power supply to maintain time?
- **Ground Truth Answer:** The serial Real-Time Clock (DS1306) in the Purge Pump Separator Assembly.
- **Context Quote:** *"A serial Real-Time Clock (DS1306) is used within an electronics box to provide an accurate time reading. The clock utilizes a button battery (CR1025) to keep track of time when the external power supply is off."*
- **NASA Lesson:** [30901](https://llis.nasa.gov/lesson/30901) — *Button Battery Power Supply to Real Time Clock for the ISS Urine Processing Assembly (UPA) Technology Upgrades Purge Pump Separator Assembly (PPSA)*
- **Center:** `MSFC` | **Date:** `2021-11-02` | **Section:** `Driving Event` | **Category:** `Hardware & Component Subsystems`
- **Benchmark Role & Revision Note:** Clarified component role in UPA PPSA electronics box.

### Question 18: What caused the serial Real-Time Clock to reset during hot restart testing of the ISS Urine Processing Assembly electronics box?
- **Ground Truth Answer:** Staking the coin cell battery into the holder over-constrained it, preventing the spring lead from maintaining contact with the negative terminal during thermal expansion.
- **Context Quote:** *"It was determined during the investigation that staking the battery into the holder effectively over constrains the battery and may prevent the spring-loaded positive lead from pressing the battery into the negative terminal from thermal expansion effects over a large temperature range."*
- **NASA Lesson:** [30901](https://llis.nasa.gov/lesson/30901) — *Button Battery Power Supply to Real Time Clock for the ISS Urine Processing Assembly (UPA) Technology Upgrades Purge Pump Separator Assembly (PPSA)*
- **Center:** `MSFC` | **Date:** `2021-11-02` | **Section:** `Driving Event` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Replaced center metadata question with verified mechanical/thermal root cause.

### Question 19: Why did the Navigation Doppler Lidar fail during the pre-launch helium purge on the New Shepard NS-17 flight?
- **Ground Truth Answer:** Helium permeated the optical filter cavity and its high thermal conductivity reduced the filter temperature, causing a center frequency shift.
- **Context Quote:** *"Investigation revealed that helium permeation of an optical filter and the high thermal conductivity of helium was responsible for this failure."*
- **NASA Lesson:** [32201](https://llis.nasa.gov/lesson/32201) — *Considerations for Helium Environments*
- **Center:** `LARC` | **Date:** `2021-08-26` | **Section:** `Abstract` | **Category:** `Physical & Thermal Mechanics`
- **Benchmark Role & Revision Note:** Center metadata question and lesson ID removed; exact physics failure mechanism provided.

### Question 20: How many facilities were covered under the Kennedy Space Center Utility Energy Services Contract scope initiated in 2020?
- **Ground Truth Answer:** 22 facilities.
- **Context Quote:** *"The Utility Energy Services Contract scope covered 22 facilities with mechanical, electrical, HVAC and water as well as street lighting and 2-megawatt (MW) solar fields."*
- **NASA Lesson:** [30402](https://llis.nasa.gov/lesson/30402) — *Utility Energy Services Contract Lessons Learned*
- **Center:** `KSC` | **Date:** `2021-08-23` | **Section:** `Driving Event` | **Category:** `Quantitative Engineering Values`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced repeating title with exact facility count from scope of work.

### Question 21: What type of electrical connectors are used in modern spacecraft designs to prevent pin damage during mating?
- **Ground Truth Answer:** Scoop Proof MIL-DTL-38999 connectors.
- **Context Quote:** *"Use of Scoop Proof MIL-DTL-38999 connectors to prevent damaging pins during connector mating"*
- **NASA Lesson:** [30101](https://llis.nasa.gov/lesson/30101) — *Cable Harness Wiring and Connector Anomalies Caused by Induced Damage in Human Spaceflight Vehicles*
- **Center:** `NESC` | **Date:** `2021-02-28` | **Section:** `Driving Event` | **Category:** `Hardware & Component Subsystems`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced title repetition ('Induced damage') with exact military connector standard.

### Question 22: Where should copper trace layers be positioned within a flex harness tape to minimize bending strain and fatigue?
- **Ground Truth Answer:** At the neutral bending axis of the tape.
- **Context Quote:** *"Note the copper layers are not at the neutral bending axis (middle) of the tape, which causes increased strain and fatigue on the copper during bending."*
- **NASA Lesson:** [31405](https://llis.nasa.gov/lesson/31405) — *Flex Harness for Dynamic Flexing Applications*
- **Center:** `LARC` | **Date:** `2021-01-01` | **Section:** `Driving Event` | **Category:** `Design & Verification Best Practices`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced repeating title with fundamental mechanical design rule.

### Question 23: Why should glassware usage be minimized during the elemental profiling of high-purity hydrazine propellant?
- **Ground Truth Answer:** To minimize sample contamination during the analytical process.
- **Context Quote:** *"Glassware usage should be minimized in all steps of the analytical process to minimize sample contamination."*
- **NASA Lesson:** [29801](https://llis.nasa.gov/lesson/29801) — *Best Practices for the Elemental Profiling of High-Purity Hydrazine*
- **Center:** `NESC` | **Date:** `2020-12-14` | **Section:** `Recommendation` | **Category:** `Materials & Specifications`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced title repetition with laboratory protocol rationale.

### Question 24: Why does NASA recommend aerodynamically stable spacecraft designs for crewed reentry?
- **Ground Truth Answer:** To enable greater robustness and not require computer control to safely return crew to Earth in the event of software errors.
- **Context Quote:** *"NASA should explicitly express a preference for stable designs that enable greater robustness and to not require computer control to safely return crew to Earth."*
- **NASA Lesson:** [33205](https://llis.nasa.gov/lesson/33205) — *Mitigating Possible Software Errors During Reentry of Crewed Spacecraft*
- **Center:** `NESC` | **Date:** `2020-12-11` | **Section:** `Abstract` | **Category:** `System Safety & Risk Management`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced title repetition with life-critical architecture principle.

### Question 25: What failure occurred during testing of a pinched copper tube for the MSolo Calibration Gas System?
- **Ground Truth Answer:** A pinch that was seemingly acceptable later separated to form a leak path.
- **Context Quote:** *"A CT scan revealed that a pinch that was seemingly acceptable later separated to form a leak path."*
- **NASA Lesson:** [29103](https://llis.nasa.gov/lesson/29103) — *Copper Tube Pinch Failure*
- **Center:** `KSC` | **Date:** `2020-10-17` | **Section:** `Driving Event` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Reformulated per reviewer feedback; no lesson ID, exact MSolo CGS failure mode.

### Question 26: Why does detailed NASA-developed simulation provide deeper government insight for programs using risk-based IV&V?
- **Ground Truth Answer:** Detailed NASA-developed/supported simulation of key flight phases provides deeper government insight and certification ability.
- **Context Quote:** *"detailed NASA-developed/supported simulation of key flight phases provides deeper government insight and certification ability."*
- **NASA Lesson:** [31901](https://llis.nasa.gov/lesson/31901) — *For NASA programs using risk-based independent verification and validation (IV&V), detailed NASA-developed/supported simulation of key flight phases provides deeper government insight and certification ability.*
- **Center:** `NESC` | **Date:** `2020-08-02` | **Section:** `Title` | **Category:** `Software & Autonomous Systems`
- **Benchmark Role & Revision Note:** Self-contained software assurance principle; praised by reviewer.

### Question 27: What was the root cause of the redundancy architecture flaw identified in the Launch Release Subsystem on the Mobile Launcher?
- **Ground Truth Answer:** The redundancy and failure analysis only reviewed interfaces with end items and did not consider potential failure scenarios of other interfacing subsystems.
- **Context Quote:** *"The root cause of the problem was that the redundancy and failure analysis for LRS was incomplete. The analysis only reviewed how LRS interfaced with the end items (umbilical release mechanisms, S&A devices, etc) and did not consider potential failure scenarios of other subsystems that interface with LRS."*
- **NASA Lesson:** [28101](https://llis.nasa.gov/lesson/28101) — *End-to-End Redundancy Analysis*
- **Center:** `KSC` | **Date:** `2020-07-14` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced title repetition with exact engineering root cause.

### Question 28: What dependency issue occurred between the RADCC and JIC audiovisual systems during verification testing at KSC?
- **Ground Truth Answer:** If one system malfunctioned or its source was unavailable, the other system was unable to provide a video source to display.
- **Context Quote:** *"If one of the systems malfunctions or its source is unavailable, the other is unable to provide a video source to display."*
- **NASA Lesson:** [29702](https://llis.nasa.gov/lesson/29702) — *Integration and Dependency Between Different Audiovisual Systems*
- **Center:** `KSC` | **Date:** `2020-03-15` | **Section:** `Driving Event` | **Category:** `System Safety & Risk Management`
- **Benchmark Role & Revision Note:** Lesson ID removed; replaced title repetition with exact operational dependency failure.

### Question 29: What hazard was identified in latching circuits using D Flip-Flops in safety-critical pyrotechnic control circuitry?
- **Ground Truth Answer:** Sensitivity to noise on the clock line that could lead to inadvertent firing.
- **Context Quote:** *"In recent designs of safety-critical pyro control circuitry, latching circuits, used to store the state of control signals, have been found to have sensitivity to noise that could lead to inadvertent firing."*
- **NASA Lesson:** [27003](https://llis.nasa.gov/lesson/27003) — *Latching Safety Critical Signals in Pyro Circuits*
- **Center:** `NESC` | **Date:** `2019-12-31` | **Section:** `Abstract` | **Category:** `Hardware & Component Subsystems`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact circuit vulnerability identified.

### Question 30: What caused the vertical and horizontal touchdown velocity errors during the Mars Science Laboratory rover landing?
- **Ground Truth Answer:** Errors in local gravity estimates.
- **Context Quote:** *"Post-landing analysis of the Mars Science Laboratory (MSL) rover touchdown revealed both vertical and horizontal velocity errors attributable to errors in local gravity estimates."*
- **NASA Lesson:** [27901](https://llis.nasa.gov/lesson/27901) — *Mars Landings May Not Accurately Account for Local Gravity (Export Version)*
- **Center:** `JPL` | **Date:** `2019-11-03` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Disambiguated per reviewer critique by specifying MSL Curiosity rover touchdown velocity errors.

### Question 31: What was found to be the cause of the weld failure on the Europa Clipper VFM flight units?
- **Ground Truth Answer:** The weld test samples used to qualify the weld procedure did not adequately represent the geometry of the flight units.
- **Context Quote:** *"Upon further investigation, it was determined that the weld test samples used to qualify the weld procedure did not adequately represent the geometry of the flight units."*
- **NASA Lesson:** [28201](https://llis.nasa.gov/lesson/28201) — *Weld Parameter Invalidation as a Result of Early Life Cycle Qualification*
- **Center:** `MSFC` | **Date:** `2019-05-12` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Exact question and answer adopted from reviewer suggestion. Note: Paired with Q32 to test RAG differentiation between twin Europa Clipper VFM lessons.

### Question 32: To what size was the radius of the Europa Clipper VFM sense tube changed without notifying the stress analysis team?
- **Ground Truth Answer:** From 0.05 inches to 0.02 inches max.
- **Context Quote:** *"The dimension of a radius in the VFM’s sense tube was changed from “0.05 inches” to “0.02 inches max” (see Figure 1)."*
- **NASA Lesson:** [28202](https://llis.nasa.gov/lesson/28202) — *Tube Stub Weld Design Change*
- **Center:** `MSFC` | **Date:** `2019-05-02` | **Section:** `Driving Event` | **Category:** `Quantitative Engineering Values`
- **Benchmark Role & Revision Note:** Exact question and answer adopted from reviewer suggestion. Note: Paired with Q31 to test RAG differentiation between twin Europa Clipper VFM lessons.

### Question 33: What was identified as the proximate cause of the SMAP radar failure on orbit?
- **Ground Truth Answer:** An unexpected over-voltage condition in a radar Low Voltage Power Supply circuit that triggered power MOSFET device sensitivity to radiation-induced damage from Single Event Effects.
- **Context Quote:** *"subsequent investigation has now identified a single proximate cause — an unexpected over-voltage condition in a radar Low Voltage Power Supply (LVPS) circuit that triggered (known) power MOSFET device sensitivity to radiation-induced damage from Single Event Effects (SEEs)."*
- **NASA Lesson:** [27701](https://llis.nasa.gov/lesson/27701) — *Proximate Cause of the SMAP Radar Failure*
- **Center:** `JPL` | **Date:** `2019-03-30` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact electronic and radiation root cause from Abstract.

### Question 34: In NASA lesson 5437, what was identified as the primary proximate cause of the Building 1100 fire at Stennis Space Center?
- **Ground Truth Answer:** An undersized extension cord that was improperly routed and energized.
- **Context Quote:** *"The primary proximate cause was an undersized extension cord that was improperly routed and energized, in the presents of fuels sources and adequate oxygen."*
- **NASA Lesson:** [5437](https://llis.nasa.gov/lesson/5437) — *Building 1100 3rd Floor Fire*
- **Center:** `SSC` | **Date:** `2019-03-13` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Lesson ID retained as BM25 benchmark test query (1 of 2 in dataset).

### Question 35: Why were results from the SLS launch availability analysis questioned by the ESD/EGS Program?
- **Ground Truth Answer:** The scope of deliverables was limited to just the analysis results, leading to limited understanding and lack of official model documentation.
- **Context Quote:** *"With the scope of deliverables limited to just the analysis results, there was limited understanding of the model upon which the analyses were accomplished. Additionally, little in the way of official model documentation was required or produced"*
- **NASA Lesson:** [25509](https://llis.nasa.gov/lesson/25509) — *Contracting for Modeling and Simulation (M&S)-based Analytical Services with Only the Analysis Results as Deliverables*
- **Center:** `NESC` | **Date:** `2019-03-08` | **Section:** `Driving Event` | **Category:** `System Safety & Risk Management`
- **Benchmark Role & Revision Note:** Disambiguated per reviewer critique by tying directly to SLS launch availability analysis under ESD/EGS.

### Question 36: What navigation sensors provide data to the NASA Autonomous Flight Termination System for destruct decisions?
- **Ground Truth Answer:** Redundant Global Positioning System/Inertial Measurement Unit (GPS/IMU) navigation sensors.
- **Context Quote:** *"The AFTS autonomously makes flight termination/destruct decisions using configurable, software-based rules that are implemented on redundant flight processors using data from redundant Global Positioning System/Inertial Measurement Unit (GPS/IMU) navigation sensors."*
- **NASA Lesson:** [24503](https://llis.nasa.gov/lesson/24503) — *Static Software Analysis of the NASA Autonomous Flight Termination Software*
- **Center:** `NESC` | **Date:** `2018-08-23` | **Section:** `Abstract` | **Category:** `Software & Autonomous Systems`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact navigation instrumentation cited.

### Question 37: Which system development model forms the primary basis for NASA policy documents NPR 7120.5 and NPR 7123.1B?
- **Ground Truth Answer:** The waterfall development model.
- **Context Quote:** *"NASA policy (NPR 7120.5 and NPR 7123.1B) and guidance (NASA Systems Engineering Handbook (SP-2016-6105), Rev 2) are primarily based on the waterfall development model."*
- **NASA Lesson:** [24502](https://llis.nasa.gov/lesson/24502) — *Aligning System Development Models with Insight Approaches*
- **Center:** `NESC` | **Date:** `2018-08-23` | **Section:** `Driving Event` | **Category:** `Standards & Compliance`
- **Benchmark Role & Revision Note:** Disambiguated per reviewer critique by citing specific NPR procedural requirements.

### Question 38: Why did the Interface Adapter Module for SAGE III on ISS suffer a failure during EMI/EMC testing?
- **Ground Truth Answer:** An impedance compatibility problem existed between the facility and the IAM, causing the supply voltage to oscillate and damage flight hardware.
- **Context Quote:** *"Investigation revealed that an impedance compatibility problem existed between the facility and the IAM. This resulted in the supply voltage oscillating to the point of damaging the IAM."*
- **NASA Lesson:** [30301](https://llis.nasa.gov/lesson/30301) — *System / Facility Impedance Checks*
- **Center:** `LARC` | **Date:** `2018-07-18` | **Section:** `Abstract` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact power distribution oscillation root cause.

### Question 39: What contaminant caused check valves on the Juno propulsion subsystem to stick closed prior to the period reduction maneuver?
- **Ground Truth Answer:** Vapor fuel-oxidizer reaction products (FORP).
- **Context Quote:** *"The most likely cause for the pre-PRM fuel check valve anomaly was contamination, with the contaminant consisting of vapor fuel-oxidizer reaction products (FORP)."*
- **NASA Lesson:** [28105](https://llis.nasa.gov/lesson/28105) — *High Oxidizer Vapor Content May Cause Vapor Reaction in Bi-Propellant Systems*
- **Center:** `JPL` | **Date:** `2018-06-19` | **Section:** `Driving Event` | **Category:** `Root Cause Analysis`
- **Benchmark Role & Revision Note:** Replaced title repetition per reviewer critique with exact chemical reaction product contaminant.

### Question 40: On which flight payload was system-level Power Quality testing for an ISS payload performed for the first time at JPL?
- **Ground Truth Answer:** The ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) flight payload.
- **Context Quote:** *"System-level PQ testing for an International Space Station (ISS) payload was performed for the first time at the NASA/Caltech Jet Propulsion Laboratory (JPL) on the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) flight payload."*
- **NASA Lesson:** [26303](https://llis.nasa.gov/lesson/26303) — *Implementing Power Quality Test Capability for ISS Payloads*
- **Center:** `JPL` | **Date:** `2018-05-30` | **Section:** `Abstract` | **Category:** `Hardware & Component Subsystems`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact flight instrument named.

### Question 41: What bearing tester was designed and built at Marshall Space Flight Center to evaluate Space Shuttle turbopump bearings?
- **Ground Truth Answer:** The Bearing, Seals, and Materials Tester (BSMT).
- **Context Quote:** *"A bearing tester, Bearing, Seals, and Materials Tester (BSMT), was designed and built at Marshall Space Flight Center (MSFC)."*
- **NASA Lesson:** [24006](https://llis.nasa.gov/lesson/24006) — *Design Guidelines for Bearings Used in Cryogenic Turbopumps and Test Rigs*
- **Center:** `MSFC` | **Date:** `2018-05-10` | **Section:** `Driving Event` | **Category:** `Hardware & Component Subsystems`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact specialized MSFC test rig named.

### Question 42: What organizational practice was identified as critical for resolving communication issues between civil servants and contractors in the EGS Program Office?
- **Ground Truth Answer:** Co-location of the civil servant and contractor teams.
- **Context Quote:** *"Several managers identified co-location of the civil servant and contractor teams as critical in addressing communication issues that impeded progress."*
- **NASA Lesson:** [25304](https://llis.nasa.gov/lesson/25304) — *Evolution of KSC EGS Post Space Shuttle - Success Factors and Lessons Learned*
- **Center:** `HQ` | **Date:** `2018-05-01` | **Section:** `Lesson Learned` | **Category:** `Design & Verification Best Practices`
- **Benchmark Role & Revision Note:** Lesson ID removed; concrete management practice from Lesson Learned.

### Question 43: What type of robot was being designed when a lithium-ion battery fire occurred at JPL in 2015?
- **Ground Truth Answer:** A multi-limbed robot.
- **Context Quote:** *"In 2015 JPL completed a non-NASA sponsored research task to design a multi-limbed robot"*
- **NASA Lesson:** [23701](https://llis.nasa.gov/lesson/23701) — *Robot Li-Ion Battery Fire*
- **Center:** `JPL` | **Date:** `2018-01-09` | **Section:** `Driving Event` | **Category:** `Hardware & Component Subsystems`
- **Benchmark Role & Revision Note:** Self-contained robotics hardware incident; no lesson ID.

### Question 44: According to NASA lesson 23201, when was the RapidScat scatterometer installed on the International Space Station?
- **Ground Truth Answer:** In September 2014.
- **Context Quote:** *"The RapidScat scatterometer was installed on the ISS in September 2014 to provide a replacement for the wide-swath scatterometry measurements that were lost when the antenna on the SeaWinds payload aboard the QuikSCAT satellite stopped rotating."*
- **NASA Lesson:** [23201](https://llis.nasa.gov/lesson/23201) — *RapidScat Power Subsystem Failure*
- **Center:** `JPL` | **Date:** `2017-12-12` | **Section:** `Driving Event` | **Category:** `Temporal Facts & Milestones`
- **Benchmark Role & Revision Note:** Lesson ID retained as BM25 benchmark test query (2 of 2 in dataset).

### Question 45: What radical operational concept was developed to recover the CloudSat mission after severe battery degradation?
- **Ground Truth Answer:** Daylight-Only Operations (DO-Op).
- **Context Quote:** *"This required developing a radical new operational concept of “Daylight-Only Operations (DO-Op).” Recovery of the CloudSat mission was made possible by the extraordinary dedication of an extraordinarily capable extended project team."*
- **NASA Lesson:** [22502](https://llis.nasa.gov/lesson/22502) — *CloudSat Battery Anomaly*
- **Center:** `JPL` | **Date:** `2017-10-03` | **Section:** `Abstract` | **Category:** `System Safety & Risk Management`
- **Benchmark Role & Revision Note:** Generic radar question replaced per reviewer critique with exact breakthrough operational concept.

### Question 46: When did the Mars Science Laboratory Curiosity rover land on the Martian surface?
- **Ground Truth Answer:** On August 6, 2012.
- **Context Quote:** *"rover landed on August 6, 2012"*
- **NASA Lesson:** [22401](https://llis.nasa.gov/lesson/22401) — *Premature Wear of the MSL Wheels*
- **Center:** `JPL` | **Date:** `2017-09-26` | **Section:** `Driving Event` | **Category:** `Temporal Facts & Milestones`
- **Benchmark Role & Revision Note:** Precise landing date; praised by reviewer.

### Question 47: What was the mission objective of the Ranger series of impact probes designed by JPL?
- **Ground Truth Answer:** To fly to the moon and intentionally collide with it to return images of the lunar surface.
- **Context Quote:** *"The Ranger series of impact probes were designed by the JPL to fly to the moon and intentionally collide with it; the objective was to return images of the surface that would aid in selection of possible Apollo landing sites."*
- **NASA Lesson:** [21501](https://llis.nasa.gov/lesson/21501) — *Projects Benefit from Lessons Learned on Previous Spaceflight Projects*
- **Center:** `JPL` | **Date:** `2017-05-23` | **Section:** `Driving Event` | **Category:** `Design & Verification Best Practices`
- **Benchmark Role & Revision Note:** Answer trimmed per reviewer suggestion to remove excess text while preserving complete fact.

### Question 48: Which Mars Phoenix collaboration tool was decommissioned upon mission completion but preserved by a developer on his personal computer?
- **Ground Truth Answer:** Sol Runner.
- **Context Quote:** *"Sol Runner, a collaboration tool used for Mars Phoenix operations, was decommissioned upon project completion. Years later, when the tool contents were needed by another Mars lander project, it was found that a Sol Runner developer had taken the initiative to save a copy on his personal computer, and the data was consequently available for restoration and reuse."*
- **NASA Lesson:** [21101](https://llis.nasa.gov/lesson/21101) — *Preserve End-of-Mission Knowledge*
- **Center:** `JPL` | **Date:** `2017-05-03` | **Section:** `Abstract` | **Category:** `Software & Autonomous Systems`
- **Benchmark Role & Revision Note:** Lesson ID removed; exact mission operations collaboration software named.

### Question 49: How many missions did the Jet Propulsion Laboratory have launched and in the operations phase when assessing its engineering checks and balances?
- **Ground Truth Answer:** An unprecedented 33 missions.
- **Context Quote:** *"JPL is the lead NASA field center for the robotic exploration of the solar system (and beyond); currently, it has an unprecedented 33 missions that have been launched and are in the operations phase"*
- **NASA Lesson:** [22601](https://llis.nasa.gov/lesson/22601) — *Employ JPL’s Established Set of Engineering ‘Checks and Balances’*
- **Center:** `JPL` | **Date:** `2017-10-24` | **Section:** `Driving Event` | **Category:** `Quantitative Engineering Values`
- **Benchmark Role & Revision Note:** Lesson ID removed; self-contained operational count.

### Question 50: Which NASA program performed integrated verification and validation testing leading up to Exploration Mission 1?
- **Ground Truth Answer:** The Exploration Ground Systems (EGS) Program.
- **Context Quote:** *"The EGS Program performed subsystem-level and integrated verification and validation (V&V) testing during its design and development phase leading up to Exploration Mission 1."*
- **NASA Lesson:** [23901](https://llis.nasa.gov/lesson/23901) — *Decision Making and Support Processes Enabling Rapid Testing*
- **Center:** `KSC` | **Date:** `2018-02-01` | **Section:** `Driving Event` | **Category:** `Standards & Compliance`
- **Benchmark Role & Revision Note:** Concrete NASA program name; praised by reviewer.

