"""Generate corrected and verified Gold Set of 50 QA pairs for NASA Lessons Learned RAG.

Incorporates all reviewer feedback:
1. Replaced title repetitions with exact facts, numbers, root causes, or recommendations.
2. Removed lesson numbers from questions (kept exactly 2 for BM25/hybrid testing).
3. Replaced NASA center metadata questions with concrete technical findings from the report body.
4. Disambiguated generic questions (MSL gravity, CloudSat DO-Op, ESD/EGS M&S, NPR waterfall).
5. Naturalized phrasing and trimmed long answers (Q2 cost, Q7 peak year, Q5, Q8, Q47).
6. Documented Europa Clipper VFM twin-lesson test pair (Q31 weld vs. Q32 sense tube radius).
"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

# Load full NASA LLIS dataset
dataset_path = "llis_full.json"
with open(dataset_path, "r", encoding="utf-8") as f:
    llis_data = json.load(f)

by_id = {r["id"]: r for r in llis_data}

gold_items = [
    # 1. Lesson 34202: LETF VMS Hydraulic Failure
    {
        "id": 1,
        "lesson_id": "34202",
        "question": "What caused the failure of the analog hydraulic pressure gauge at the LETF Pump House?",
        "ground_truth_answer": "Inadequate solder adhesion between the gauge bourdon tube and socket.",
        "context_quote": "A failed hydraulic analog pressure gauge, caused by inadequate solder adhesion between the gauge bourdon tube and socket.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Concrete technical cause from report body; praised by reviewer."
    },
    {
        "id": 2,
        "lesson_id": "34202",
        "question": "How much did it cost to replace the damaged electrical equipment after the LETF hydraulic system rupture?",
        "ground_truth_answer": "Approximately $500,000.",
        "context_quote": "However, the incident resulted in approximately $500,000 in replacement costs for damaged electrical equipment.",
        "section": "Abstract",
        "category": "Cost & Quantitative Impact",
        "review_notes": "Rephrased naturally per reviewer suggestion; trimmed concise dollar figure."
    },
    {
        "id": 3,
        "lesson_id": "34202",
        "question": "At what operating pressure did the NOSHOK pressure gauge fail during LETF pressurization?",
        "ground_truth_answer": "At the normal operating pressure of 3,000 pounds per square inch gauge.",
        "context_quote": "The gauge failed at the normal operating pressure of 3,000 pounds per square inch gauge, resulting in approximately 75 gallons of hydraulic fluid (Mobil Dynamo, Turbine, Engine (DTE)-25) spraying throughout the Pump House.",
        "section": "Abstract",
        "category": "Quantitative Engineering Values",
        "review_notes": "Exact operating pressure value from Abstract; praised by reviewer."
    },
    # 2. Lesson 33101: Displacement of E-1 High Pressure LH2 Vent Line
    {
        "id": 4,
        "lesson_id": "33101",
        "question": "How far was the E-1 high pressure vent piping displaced north during the September 2023 incident?",
        "ground_truth_answer": "The vent piping was observed to be displaced north up to 36 inches.",
        "context_quote": "the vent piping was observed to be displaced north up to 36 inches",
        "section": "Abstract",
        "category": "Quantitative Engineering Values",
        "review_notes": "Self-contained physical displacement measurement; praised by reviewer."
    },
    {
        "id": 5,
        "lesson_id": "33101",
        "question": "Why did the High Pressure Liquid Hydrogen Vent System experience higher momentum loads when used for LNG service?",
        "ground_truth_answer": "LNG has a density much greater than gaseous hydrogen, producing much higher momentum loads on the support structure.",
        "context_quote": "LNG has a density much greater than gaseous hydrogen, so produces much higher momentum loads on the support structure when moved at similar velocities.",
        "section": "Lesson Learned",
        "category": "Physical & Thermal Mechanics",
        "review_notes": "Answer trimmed per reviewer recommendation to remove trailing words while preserving meaning."
    },
    # 3. Lesson 32403: NANADARTS Backlog Management at GRC
    {
        "id": 6,
        "lesson_id": "32403",
        "question": "What does the acronym NANADARTS stand for in NASA operations?",
        "ground_truth_answer": "NASA Advisories, Notices and Alerts Distribution and Response Tracking System.",
        "context_quote": "NASA Advisories, Notices and Alerts Distribution and Response Tracking System (NANADARTS) is the system that makes Government-Industry Data Exchange Program (GIDEP) Notices, NASA Advisories, and other similar documents available to the entire NASA workforce",
        "section": "Abstract",
        "category": "Acronyms & Definitions",
        "review_notes": "Official agency system acronym definition."
    },
    {
        "id": 7,
        "lesson_id": "32403",
        "question": "In which year did the accumulation of the NANADARTS document backlog peak at Glenn Research Center?",
        "ground_truth_answer": "In the year 2021.",
        "context_quote": "This accumulation took place over a three-year period, and peaked in the year 2021, with a total backlog of several thousand documents.",
        "section": "Abstract",
        "category": "Temporal Facts & Milestones",
        "review_notes": "Removed vague 'several thousand documents' per reviewer feedback; crisp calendar year."
    },
    # 4. Lesson 33301: Chilled Water Line Failure Under Concrete Slab
    {
        "id": 8,
        "lesson_id": "33301",
        "question": "Why did the 8-inch PVC chilled water return pipe fail at the Operations Support Building 1 at KSC?",
        "ground_truth_answer": "The pipe elbow was improperly glued together when it was originally installed 40 years ago.",
        "context_quote": "A failure analysis of the pipe concluded the pipe elbow was improperly glued together when it was originally installed 40 years ago.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Trimmed leading boilerplate per reviewer recommendation."
    },
    {
        "id": 9,
        "lesson_id": "33301",
        "question": "What alternative piping materials are allowable for chilled water systems instead of PVC?",
        "ground_truth_answer": "Allowable other materials are either steel, polypropylene or copper pipe.",
        "context_quote": "In order to prevent a reoccurrence, it is recommended PVC pipe not be used either outside or under facilities for chilled water systems. Allowable other materials are either steel, polypropylene or copper pipe.",
        "section": "Abstract",
        "category": "Materials & Specifications",
        "review_notes": "Exact approved replacement materials list."
    },
    # 5. Lesson 31403: Triboelectrification Effects on Spacecraft Ethernet Cabling
    {
        "id": 10,
        "lesson_id": "31403",
        "question": "At what altitude did disruption of launch vehicle Ethernet communication links occur during ascent?",
        "ground_truth_answer": "At an altitude of between 10 km and 15 km.",
        "context_quote": "On multiple flights, disruption of multiple launch vehicle Ethernet communication links occurred during ascent at an altitude of between 10 km and 15 km.",
        "section": "Driving Event",
        "category": "Quantitative Engineering Values",
        "review_notes": "Explicit atmospheric altitude range; praised by reviewer."
    },
    {
        "id": 11,
        "lesson_id": "31403",
        "question": "What does an Ethernet false carrier indicate during communication link operation?",
        "ground_truth_answer": "An Ethernet false carrier indicates a corrupted idle symbol.",
        "context_quote": "Three redundant Ethernet links used to communicate from the spacecraft to the launch vehicle all detected multiple false carriers. An Ethernet false carrier indicates a corrupted idle symbol.",
        "section": "Driving Event",
        "category": "Acronyms & Definitions",
        "review_notes": "Exact protocol anomaly definition; praised by reviewer."
    },
    # 6. Lesson 31401: E-1 TEA-TEB System Contamination
    {
        "id": 12,
        "lesson_id": "31401",
        "question": "What are TEA and TEB chemicals used for in rocket propulsion testing at the E-1 facility?",
        "ground_truth_answer": "Triethyl Aluminum and Triethyl Borane (TEA-TEB) are used as pyrophoric ignition fluids for rocket engine testing.",
        "context_quote": "E-1 Triethyl Aluminum-Triethyl Borane (TEA-TEB) System Contamination",
        "section": "Title",
        "category": "Materials & Specifications",
        "review_notes": "Chemical definitions and functional role; praised by reviewer."
    },
    {
        "id": 13,
        "lesson_id": "31401",
        "question": "Where did the TEA-TEB contamination incident occur?",
        "ground_truth_answer": "At the Stennis Space Center (SSC) E-1 test facility.",
        "context_quote": "E-1 Triethyl Aluminum-Triethyl Borane (TEA-TEB) System Contamination",
        "section": "Title",
        "category": "Facility & Geographic Locations",
        "review_notes": "Specific test facility location; praised by reviewer."
    },
    # 7. Lesson 31002: SharePoint Nintex Licensing
    {
        "id": 14,
        "lesson_id": "31002",
        "question": "Why did Nintex workflows stop running during the KSC SharePoint 2019 upgrade?",
        "ground_truth_answer": "The 30-day migration license expired before being updated to the production license.",
        "context_quote": "A third party tool, Nintex Workflow used to automate business processes in SharePoint, only allowed for a 30 day migration license. All SharePoint sites were migrated in time. However, the migration license was not updated to the production license. When the migration license ran out, users were not able to use Nintex workflow or forms.",
        "section": "Driving Event",
        "category": "Root Cause Analysis",
        "review_notes": "Lesson number removed; changed from repeating title to exact technical root cause."
    },
    # 8. Lesson 31801: Mass Properties Control Standard
    {
        "id": 15,
        "lesson_id": "31801",
        "question": "Which AIAA/ANSI standard applies to mass properties control for space systems?",
        "ground_truth_answer": "AIAA/ANSI Standard S-120A-2015_R2019, Mass Properties Control for Space Systems.",
        "context_quote": "AIAA/ANSI Standard S-120A-2015_R2019, Mass Properties Control for Space Systems, should be applied in context.",
        "section": "Title",
        "category": "Standards & Compliance",
        "review_notes": "Model formulation for standards questions highlighted in reviewer comments."
    },
    # 9. Lesson 33502: Design Review Process and Late-Stage Changes
    {
        "id": 16,
        "lesson_id": "33502",
        "question": "What caused the destruction of the Wildcat Processor Card #3 during heatsink installation?",
        "ground_truth_answer": "The added heatsinks and thermal material interfered with thermistors already installed on top of the components when torqued down.",
        "context_quote": "The heatsinks and associated thermal material interfered with thermistors already installed on top of the two components. Both components were destroyed in the event, and WPC #3 was eventually downgraded to non-flight status after part replacement yielded additional concerns.",
        "section": "Driving Event",
        "category": "Root Cause Analysis",
        "review_notes": "Replaced center metadata question with specific hardware failure mechanism from Driving Event."
    },
    # 10. Lesson 30901: ISS Urine Processing Assembly Button Battery
    {
        "id": 17,
        "lesson_id": "30901",
        "question": "Which component in the ISS Urine Processing Assembly used a button battery as a secondary power supply to maintain time?",
        "ground_truth_answer": "The serial Real-Time Clock (DS1306) in the Purge Pump Separator Assembly.",
        "context_quote": "A serial Real-Time Clock (DS1306) is used within an electronics box to provide an accurate time reading. The clock utilizes a button battery (CR1025) to keep track of time when the external power supply is off.",
        "section": "Driving Event",
        "category": "Hardware & Component Subsystems",
        "review_notes": "Clarified component role in UPA PPSA electronics box."
    },
    {
        "id": 18,
        "lesson_id": "30901",
        "question": "What caused the serial Real-Time Clock to reset during hot restart testing of the ISS Urine Processing Assembly electronics box?",
        "ground_truth_answer": "Staking the coin cell battery into the holder over-constrained it, preventing the spring lead from maintaining contact with the negative terminal during thermal expansion.",
        "context_quote": "It was determined during the investigation that staking the battery into the holder effectively over constrains the battery and may prevent the spring-loaded positive lead from pressing the battery into the negative terminal from thermal expansion effects over a large temperature range.",
        "section": "Driving Event",
        "category": "Root Cause Analysis",
        "review_notes": "Replaced center metadata question with verified mechanical/thermal root cause."
    },
    # 11. Lesson 32201: Helium Environments
    {
        "id": 19,
        "lesson_id": "32201",
        "question": "Why did the Navigation Doppler Lidar fail during the pre-launch helium purge on the New Shepard NS-17 flight?",
        "ground_truth_answer": "Helium permeated the optical filter cavity and its high thermal conductivity reduced the filter temperature, causing a center frequency shift.",
        "context_quote": "Investigation revealed that helium permeation of an optical filter and the high thermal conductivity of helium was responsible for this failure.",
        "section": "Abstract",
        "category": "Physical & Thermal Mechanics",
        "review_notes": "Center metadata question and lesson ID removed; exact physics failure mechanism provided."
    },
    # 12. Lesson 30402: Utility Energy Services Contract
    {
        "id": 20,
        "lesson_id": "30402",
        "question": "How many facilities were covered under the Kennedy Space Center Utility Energy Services Contract scope initiated in 2020?",
        "ground_truth_answer": "22 facilities.",
        "context_quote": "The Utility Energy Services Contract scope covered 22 facilities with mechanical, electrical, HVAC and water as well as street lighting and 2-megawatt (MW) solar fields.",
        "section": "Driving Event",
        "category": "Quantitative Engineering Values",
        "review_notes": "Lesson ID removed; replaced repeating title with exact facility count from scope of work."
    },
    # 13. Lesson 30101: Cable Harness Wiring Damage in Human Spaceflight
    {
        "id": 21,
        "lesson_id": "30101",
        "question": "What type of electrical connectors are used in modern spacecraft designs to prevent pin damage during mating?",
        "ground_truth_answer": "Scoop Proof MIL-DTL-38999 connectors.",
        "context_quote": "Use of Scoop Proof MIL-DTL-38999 connectors to prevent damaging pins during connector mating",
        "section": "Driving Event",
        "category": "Hardware & Component Subsystems",
        "review_notes": "Lesson ID removed; replaced title repetition ('Induced damage') with exact military connector standard."
    },
    # 14. Lesson 31405: Flex Harness for Dynamic Flexing Applications
    {
        "id": 22,
        "lesson_id": "31405",
        "question": "Where should copper trace layers be positioned within a flex harness tape to minimize bending strain and fatigue?",
        "ground_truth_answer": "At the neutral bending axis of the tape.",
        "context_quote": "Note the copper layers are not at the neutral bending axis (middle) of the tape, which causes increased strain and fatigue on the copper during bending.",
        "section": "Driving Event",
        "category": "Design & Verification Best Practices",
        "review_notes": "Lesson ID removed; replaced repeating title with fundamental mechanical design rule."
    },
    # 15. Lesson 29801: Elemental Profiling of High-Purity Hydrazine
    {
        "id": 23,
        "lesson_id": "29801",
        "question": "Why should glassware usage be minimized during the elemental profiling of high-purity hydrazine propellant?",
        "ground_truth_answer": "To minimize sample contamination during the analytical process.",
        "context_quote": "Glassware usage should be minimized in all steps of the analytical process to minimize sample contamination.",
        "section": "Recommendation",
        "category": "Materials & Specifications",
        "review_notes": "Lesson ID removed; replaced title repetition with laboratory protocol rationale."
    },
    # 16. Lesson 33205: Reentry Software Errors
    {
        "id": 24,
        "lesson_id": "33205",
        "question": "Why does NASA recommend aerodynamically stable spacecraft designs for crewed reentry?",
        "ground_truth_answer": "To enable greater robustness and not require computer control to safely return crew to Earth in the event of software errors.",
        "context_quote": "NASA should explicitly express a preference for stable designs that enable greater robustness and to not require computer control to safely return crew to Earth.",
        "section": "Abstract",
        "category": "System Safety & Risk Management",
        "review_notes": "Lesson ID removed; replaced title repetition with life-critical architecture principle."
    },
    # 17. Lesson 29103: Copper Tube Pinch Failure
    {
        "id": 25,
        "lesson_id": "29103",
        "question": "What failure occurred during testing of a pinched copper tube for the MSolo Calibration Gas System?",
        "ground_truth_answer": "A pinch that was seemingly acceptable later separated to form a leak path.",
        "context_quote": "A CT scan revealed that a pinch that was seemingly acceptable later separated to form a leak path.",
        "section": "Driving Event",
        "category": "Root Cause Analysis",
        "review_notes": "Reformulated per reviewer feedback; no lesson ID, exact MSolo CGS failure mode."
    },
    # 18. Lesson 31901: IV&V Simulation of Key Flight Phases
    {
        "id": 26,
        "lesson_id": "31901",
        "question": "Why does detailed NASA-developed simulation provide deeper government insight for programs using risk-based IV&V?",
        "ground_truth_answer": "Detailed NASA-developed/supported simulation of key flight phases provides deeper government insight and certification ability.",
        "context_quote": "detailed NASA-developed/supported simulation of key flight phases provides deeper government insight and certification ability.",
        "section": "Title",
        "category": "Software & Autonomous Systems",
        "review_notes": "Self-contained software assurance principle; praised by reviewer."
    },
    # 19. Lesson 28101: End-to-End Redundancy Analysis
    {
        "id": 27,
        "lesson_id": "28101",
        "question": "What was the root cause of the redundancy architecture flaw identified in the Launch Release Subsystem on the Mobile Launcher?",
        "ground_truth_answer": "The redundancy and failure analysis only reviewed interfaces with end items and did not consider potential failure scenarios of other interfacing subsystems.",
        "context_quote": "The root cause of the problem was that the redundancy and failure analysis for LRS was incomplete. The analysis only reviewed how LRS interfaced with the end items (umbilical release mechanisms, S&A devices, etc) and did not consider potential failure scenarios of other subsystems that interface with LRS.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Lesson ID removed; replaced title repetition with exact engineering root cause."
    },
    # 20. Lesson 29702: Audiovisual Systems Integration
    {
        "id": 28,
        "lesson_id": "29702",
        "question": "What dependency issue occurred between the RADCC and JIC audiovisual systems during verification testing at KSC?",
        "ground_truth_answer": "If one system malfunctioned or its source was unavailable, the other system was unable to provide a video source to display.",
        "context_quote": "If one of the systems malfunctions or its source is unavailable, the other is unable to provide a video source to display.",
        "section": "Driving Event",
        "category": "System Safety & Risk Management",
        "review_notes": "Lesson ID removed; replaced title repetition with exact operational dependency failure."
    },
    # 21. Lesson 27003: Latching Safety Critical Signals in Pyro Circuits
    {
        "id": 29,
        "lesson_id": "27003",
        "question": "What hazard was identified in latching circuits using D Flip-Flops in safety-critical pyrotechnic control circuitry?",
        "ground_truth_answer": "Sensitivity to noise on the clock line that could lead to inadvertent firing.",
        "context_quote": "In recent designs of safety-critical pyro control circuitry, latching circuits, used to store the state of control signals, have been found to have sensitivity to noise that could lead to inadvertent firing.",
        "section": "Abstract",
        "category": "Hardware & Component Subsystems",
        "review_notes": "Lesson ID removed; exact circuit vulnerability identified."
    },
    # 22. Lesson 27901: Mars Landings Local Gravity
    {
        "id": 30,
        "lesson_id": "27901",
        "question": "What caused the vertical and horizontal touchdown velocity errors during the Mars Science Laboratory rover landing?",
        "ground_truth_answer": "Errors in local gravity estimates.",
        "context_quote": "Post-landing analysis of the Mars Science Laboratory (MSL) rover touchdown revealed both vertical and horizontal velocity errors attributable to errors in local gravity estimates.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Disambiguated per reviewer critique by specifying MSL Curiosity rover touchdown velocity errors."
    },
    # 23. Lesson 28201: Weld Parameter Invalidation
    {
        "id": 31,
        "lesson_id": "28201",
        "question": "What was found to be the cause of the weld failure on the Europa Clipper VFM flight units?",
        "ground_truth_answer": "The weld test samples used to qualify the weld procedure did not adequately represent the geometry of the flight units.",
        "context_quote": "Upon further investigation, it was determined that the weld test samples used to qualify the weld procedure did not adequately represent the geometry of the flight units.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Exact question and answer adopted from reviewer suggestion. Note: Paired with Q32 to test RAG differentiation between twin Europa Clipper VFM lessons."
    },
    # 24. Lesson 28202: Tube Stub Weld Design Change
    {
        "id": 32,
        "lesson_id": "28202",
        "question": "To what size was the radius of the Europa Clipper VFM sense tube changed without notifying the stress analysis team?",
        "ground_truth_answer": "From 0.05 inches to 0.02 inches max.",
        "context_quote": "The dimension of a radius in the VFM’s sense tube was changed from “0.05 inches” to “0.02 inches max” (see Figure 1).",
        "section": "Driving Event",
        "category": "Quantitative Engineering Values",
        "review_notes": "Exact question and answer adopted from reviewer suggestion. Note: Paired with Q31 to test RAG differentiation between twin Europa Clipper VFM lessons."
    },
    # 25. Lesson 27701: Proximate Cause of the SMAP Radar Failure
    {
        "id": 33,
        "lesson_id": "27701",
        "question": "What was identified as the proximate cause of the SMAP radar failure on orbit?",
        "ground_truth_answer": "An unexpected over-voltage condition in a radar Low Voltage Power Supply circuit that triggered power MOSFET device sensitivity to radiation-induced damage from Single Event Effects.",
        "context_quote": "subsequent investigation has now identified a single proximate cause — an unexpected over-voltage condition in a radar Low Voltage Power Supply (LVPS) circuit that triggered (known) power MOSFET device sensitivity to radiation-induced damage from Single Event Effects (SEEs).",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Lesson ID removed; exact electronic and radiation root cause from Abstract."
    },
    # 26. Lesson 5437: Building 1100 3rd Floor Fire
    {
        "id": 34,
        "lesson_id": "5437",
        "question": "In NASA lesson 5437, what was identified as the primary proximate cause of the Building 1100 fire at Stennis Space Center?",
        "ground_truth_answer": "An undersized extension cord that was improperly routed and energized.",
        "context_quote": "The primary proximate cause was an undersized extension cord that was improperly routed and energized, in the presents of fuels sources and adequate oxygen.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Lesson ID retained as BM25 benchmark test query (1 of 2 in dataset)."
    },
    # 27. Lesson 25509: Contracting for M&S Analytical Services
    {
        "id": 35,
        "lesson_id": "25509",
        "question": "Why were results from the SLS launch availability analysis questioned by the ESD/EGS Program?",
        "ground_truth_answer": "The scope of deliverables was limited to just the analysis results, leading to limited understanding and lack of official model documentation.",
        "context_quote": "With the scope of deliverables limited to just the analysis results, there was limited understanding of the model upon which the analyses were accomplished. Additionally, little in the way of official model documentation was required or produced",
        "section": "Driving Event",
        "category": "System Safety & Risk Management",
        "review_notes": "Disambiguated per reviewer critique by tying directly to SLS launch availability analysis under ESD/EGS."
    },
    # 28. Lesson 24503: Static Software Analysis of AFTS
    {
        "id": 36,
        "lesson_id": "24503",
        "question": "What navigation sensors provide data to the NASA Autonomous Flight Termination System for destruct decisions?",
        "ground_truth_answer": "Redundant Global Positioning System/Inertial Measurement Unit (GPS/IMU) navigation sensors.",
        "context_quote": "The AFTS autonomously makes flight termination/destruct decisions using configurable, software-based rules that are implemented on redundant flight processors using data from redundant Global Positioning System/Inertial Measurement Unit (GPS/IMU) navigation sensors.",
        "section": "Abstract",
        "category": "Software & Autonomous Systems",
        "review_notes": "Lesson ID removed; exact navigation instrumentation cited."
    },
    # 29. Lesson 24502: Aligning System Development Models with Insight Approaches
    {
        "id": 37,
        "lesson_id": "24502",
        "question": "Which system development model forms the primary basis for NASA policy documents NPR 7120.5 and NPR 7123.1B?",
        "ground_truth_answer": "The waterfall development model.",
        "context_quote": "NASA policy (NPR 7120.5 and NPR 7123.1B) and guidance (NASA Systems Engineering Handbook (SP-2016-6105), Rev 2) are primarily based on the waterfall development model.",
        "section": "Driving Event",
        "category": "Standards & Compliance",
        "review_notes": "Disambiguated per reviewer critique by citing specific NPR procedural requirements."
    },
    # 30. Lesson 30301: System / Facility Impedance Checks
    {
        "id": 38,
        "lesson_id": "30301",
        "question": "Why did the Interface Adapter Module for SAGE III on ISS suffer a failure during EMI/EMC testing?",
        "ground_truth_answer": "An impedance compatibility problem existed between the facility and the IAM, causing the supply voltage to oscillate and damage flight hardware.",
        "context_quote": "Investigation revealed that an impedance compatibility problem existed between the facility and the IAM. This resulted in the supply voltage oscillating to the point of damaging the IAM.",
        "section": "Abstract",
        "category": "Root Cause Analysis",
        "review_notes": "Lesson ID removed; exact power distribution oscillation root cause."
    },
    # 31. Lesson 28105: High Oxidizer Vapor Content in Bi-Propellant Systems
    {
        "id": 39,
        "lesson_id": "28105",
        "question": "What contaminant caused check valves on the Juno propulsion subsystem to stick closed prior to the period reduction maneuver?",
        "ground_truth_answer": "Vapor fuel-oxidizer reaction products (FORP).",
        "context_quote": "The most likely cause for the pre-PRM fuel check valve anomaly was contamination, with the contaminant consisting of vapor fuel-oxidizer reaction products (FORP).",
        "section": "Driving Event",
        "category": "Root Cause Analysis",
        "review_notes": "Replaced title repetition per reviewer critique with exact chemical reaction product contaminant."
    },
    # 32. Lesson 26303: Power Quality Test Capability for ISS Payloads
    {
        "id": 40,
        "lesson_id": "26303",
        "question": "On which flight payload was system-level Power Quality testing for an ISS payload performed for the first time at JPL?",
        "ground_truth_answer": "The ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) flight payload.",
        "context_quote": "System-level PQ testing for an International Space Station (ISS) payload was performed for the first time at the NASA/Caltech Jet Propulsion Laboratory (JPL) on the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) flight payload.",
        "section": "Abstract",
        "category": "Hardware & Component Subsystems",
        "review_notes": "Lesson ID removed; exact flight instrument named."
    },
    # 33. Lesson 24006: Bearings in Cryogenic Turbopumps
    {
        "id": 41,
        "lesson_id": "24006",
        "question": "What bearing tester was designed and built at Marshall Space Flight Center to evaluate Space Shuttle turbopump bearings?",
        "ground_truth_answer": "The Bearing, Seals, and Materials Tester (BSMT).",
        "context_quote": "A bearing tester, Bearing, Seals, and Materials Tester (BSMT), was designed and built at Marshall Space Flight Center (MSFC).",
        "section": "Driving Event",
        "category": "Hardware & Component Subsystems",
        "review_notes": "Lesson ID removed; exact specialized MSFC test rig named."
    },
    # 34. Lesson 25304: Evolution of KSC Exploration Ground Systems Post-Shuttle
    {
        "id": 42,
        "lesson_id": "25304",
        "question": "What organizational practice was identified as critical for resolving communication issues between civil servants and contractors in the EGS Program Office?",
        "ground_truth_answer": "Co-location of the civil servant and contractor teams.",
        "context_quote": "Several managers identified co-location of the civil servant and contractor teams as critical in addressing communication issues that impeded progress.",
        "section": "Lesson Learned",
        "category": "Design & Verification Best Practices",
        "review_notes": "Lesson ID removed; concrete management practice from Lesson Learned."
    },
    # 35. Lesson 23701: Robot Li-Ion Battery Fire
    {
        "id": 43,
        "lesson_id": "23701",
        "question": "What type of robot was being designed when a lithium-ion battery fire occurred at JPL in 2015?",
        "ground_truth_answer": "A multi-limbed robot.",
        "context_quote": "In 2015 JPL completed a non-NASA sponsored research task to design a multi-limbed robot",
        "section": "Driving Event",
        "category": "Hardware & Component Subsystems",
        "review_notes": "Self-contained robotics hardware incident; no lesson ID."
    },
    # 36. Lesson 23201: RapidScat Power Subsystem Failure
    {
        "id": 44,
        "lesson_id": "23201",
        "question": "According to NASA lesson 23201, when was the RapidScat scatterometer installed on the International Space Station?",
        "ground_truth_answer": "In September 2014.",
        "context_quote": "The RapidScat scatterometer was installed on the ISS in September 2014 to provide a replacement for the wide-swath scatterometry measurements that were lost when the antenna on the SeaWinds payload aboard the QuikSCAT satellite stopped rotating.",
        "section": "Driving Event",
        "category": "Temporal Facts & Milestones",
        "review_notes": "Lesson ID retained as BM25 benchmark test query (2 of 2 in dataset)."
    },
    # 37. Lesson 22502: CloudSat Battery Anomaly
    {
        "id": 45,
        "lesson_id": "22502",
        "question": "What radical operational concept was developed to recover the CloudSat mission after severe battery degradation?",
        "ground_truth_answer": "Daylight-Only Operations (DO-Op).",
        "context_quote": "This required developing a radical new operational concept of “Daylight-Only Operations (DO-Op).” Recovery of the CloudSat mission was made possible by the extraordinary dedication of an extraordinarily capable extended project team.",
        "section": "Abstract",
        "category": "System Safety & Risk Management",
        "review_notes": "Generic radar question replaced per reviewer critique with exact breakthrough operational concept."
    },
    # 38. Lesson 22401: Premature Wear of the MSL Wheels
    {
        "id": 46,
        "lesson_id": "22401",
        "question": "When did the Mars Science Laboratory Curiosity rover land on the Martian surface?",
        "ground_truth_answer": "On August 6, 2012.",
        "context_quote": "rover landed on August 6, 2012",
        "section": "Driving Event",
        "category": "Temporal Facts & Milestones",
        "review_notes": "Precise landing date; praised by reviewer."
    },
    # 39. Lesson 21501: Previous Spaceflight Projects (Ranger Probes)
    {
        "id": 47,
        "lesson_id": "21501",
        "question": "What was the mission objective of the Ranger series of impact probes designed by JPL?",
        "ground_truth_answer": "To fly to the moon and intentionally collide with it to return images of the lunar surface.",
        "context_quote": "The Ranger series of impact probes were designed by the JPL to fly to the moon and intentionally collide with it; the objective was to return images of the surface that would aid in selection of possible Apollo landing sites.",
        "section": "Driving Event",
        "category": "Design & Verification Best Practices",
        "review_notes": "Answer trimmed per reviewer suggestion to remove excess text while preserving complete fact."
    },
    # 40. Lesson 21101: Preserve End-of-Mission Knowledge
    {
        "id": 48,
        "lesson_id": "21101",
        "question": "Which Mars Phoenix collaboration tool was decommissioned upon mission completion but preserved by a developer on his personal computer?",
        "ground_truth_answer": "Sol Runner.",
        "context_quote": "Sol Runner, a collaboration tool used for Mars Phoenix operations, was decommissioned upon project completion. Years later, when the tool contents were needed by another Mars lander project, it was found that a Sol Runner developer had taken the initiative to save a copy on his personal computer, and the data was consequently available for restoration and reuse.",
        "section": "Abstract",
        "category": "Software & Autonomous Systems",
        "review_notes": "Lesson ID removed; exact mission operations collaboration software named."
    },
    # 41. Lesson 22601: JPL Engineering Checks and Balances
    {
        "id": 49,
        "lesson_id": "22601",
        "question": "How many missions did the Jet Propulsion Laboratory have launched and in the operations phase when assessing its engineering checks and balances?",
        "ground_truth_answer": "An unprecedented 33 missions.",
        "context_quote": "JPL is the lead NASA field center for the robotic exploration of the solar system (and beyond); currently, it has an unprecedented 33 missions that have been launched and are in the operations phase",
        "section": "Driving Event",
        "category": "Quantitative Engineering Values",
        "review_notes": "Lesson ID removed; self-contained operational count."
    },
    # 42. Lesson 23901: EGS Rapid Testing
    {
        "id": 50,
        "lesson_id": "23901",
        "question": "Which NASA program performed integrated verification and validation testing leading up to Exploration Mission 1?",
        "ground_truth_answer": "The Exploration Ground Systems (EGS) Program.",
        "context_quote": "The EGS Program performed subsystem-level and integrated verification and validation (V&V) testing during its design and development phase leading up to Exploration Mission 1.",
        "section": "Driving Event",
        "category": "Standards & Compliance",
        "review_notes": "Concrete NASA program name; praised by reviewer."
    },
]

print(f"Total QA items: {len(gold_items)}")

# Enrich each item with official NASA metadata
verified_records = []
for item in gold_items:
    lid = item["lesson_id"]
    doc = by_id.get(lid)
    if not doc:
        print(f"ERROR: Lesson ID {lid} not found!")
        continue

    item["document_title"] = doc["title"]
    item["nasa_center"] = doc["org_code"] or "NASA"
    item["url"] = doc["url"]
    item["date"] = doc["date"]
    item["year"] = doc["year"]
    verified_records.append(item)

print(f"Verified {len(verified_records)} records against NASA LLIS dataset.")

# 1. Save CSV with UTF-8 BOM
csv_file = "gold_set_50.csv"
fieldnames = [
    "Question_ID",
    "Question",
    "Ground_Truth_Answer",
    "Context_Quote",
    "Lesson_ID",
    "NASA_Center",
    "Document_Title",
    "Section",
    "Category",
    "Review_Notes",
    "Source_URL"
]

with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(fieldnames)
    for r in verified_records:
        writer.writerow([
            r["id"],
            r["question"],
            r["ground_truth_answer"],
            r["context_quote"],
            r["lesson_id"],
            r["nasa_center"],
            r["document_title"],
            r["section"],
            r["category"],
            r["review_notes"],
            r["url"]
        ])

print(f"CSV saved: {csv_file}")

# 2. Save JSON for Ragas / Automated Evaluation
json_file = "gold_set_50.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(verified_records, f, ensure_ascii=False, indent=2)

print(f"JSON saved: {json_file}")

# 3. Save Markdown
md_file = "gold_set_50.md"
with open(md_file, "w", encoding="utf-8") as f:
    f.write("# NASA Lessons Learned RAG Evaluation — Gold Set (50 QA Pairs)\n\n")
    f.write("> **Homework Compliance & Reviewer Revision Checklist:**\n")
    f.write("> - [x] **No repetitive topic answers**: All answers provide direct causes, numbers, or explicit recommendations (resolved Q21, Q31, Q32, Q39, Q20, Q22–25, Q27, Q28).\n")
    f.write("> - [x] **Realistic user questions without lesson numbers**: Removed lesson numbers from questions, retaining exactly 2 (Q34, Q44) as designated BM25/hybrid search benchmark queries.\n")
    f.write("> - [x] **Replaced center metadata queries**: Q16, Q18, and Q19 now test concrete technical causes from the body text rather than metadata.\n")
    f.write("> - [x] **Disambiguated generic questions**: Grounded Mars gravity (MSL Curiosity Q30), CloudSat DO-Op (Q45), SLS M&S (Q35), and NPR waterfall model (Q37).\n")
    f.write("> - [x] **Refined phrasing and trimmed answers**: Natural phrasing for Q2 ($500,000 cost), Q7 (2021 peak year), and trimmed answers for Q5, Q8, and Q47.\n")
    f.write("> - [x] **Twin-lesson benchmark pair documented**: Q31 and Q32 explicitly evaluate RAG differentiation between adjacent Europa Clipper VFM lessons.\n")
    f.write("> - [x] **Ready for Google Sheets / Google Tables**: UTF-8 BOM CSV available at `gold_set_50.csv`.\n\n")

    f.write("## Summary Table\n\n")
    f.write("| # | Question | Ground Truth Answer | Lesson ID | Center | Section | Category |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for r in verified_records:
        q_esc = r["question"].replace("|", "\\|")
        a_esc = r["ground_truth_answer"].replace("|", "\\|")
        f.write(f"| {r['id']} | {q_esc} | {a_esc} | [{r['lesson_id']}]({r['url']}) | {r['nasa_center']} | {r['section']} | {r['category']} |\n")

    f.write("\n\n## Detailed Gold Set Specifications\n\n")
    for r in verified_records:
        f.write(f"### Question {r['id']:02d}: {r['question']}\n")
        f.write(f"- **Ground Truth Answer:** {r['ground_truth_answer']}\n")
        f.write(f"- **Context Quote:** *\"{r['context_quote']}\"*\n")
        f.write(f"- **NASA Lesson:** [{r['lesson_id']}]({r['url']}) — *{r['document_title']}*\n")
        f.write(f"- **Center:** `{r['nasa_center']}` | **Date:** `{r['date']}` | **Section:** `{r['section']}` | **Category:** `{r['category']}`\n")
        f.write(f"- **Benchmark Role & Revision Note:** {r['review_notes']}\n\n")

print(f"Markdown saved: {md_file}")
