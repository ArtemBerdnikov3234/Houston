"""Generate and verify the HARD Gold Set for the NASA LLIS retriever benchmark.

Complements gold_set_50 (lessons 2010–2025, mostly Abstract/Driving Event, lexically close to the text):
1. All questions come from lessons dated 1990–2009 (~80% of the database).
2. Most answers live in Lesson Learned / Recommendation / Evidence / References sections.
3. Questions are paraphrased the way an engineer would ask, avoiding the exact terms of the report,
   so BM25 cannot win on lexical overlap alone (overlap is measured and reported per question).
4. Some questions carry a metadata filter ("filter": {"org_code": ...} or {"program": ...}).
5. Some questions are answered by several lessons ("relevant_lesson_ids"); any of them counts as a hit.

Verification (the script exits with an error if any check fails):
- context_quote is verbatim in the lesson full_text in llis_full.json and in the declared section;
- context_quote lies entirely inside one chunk of llis_chunks.jsonl whose metadata.lesson_ids contains the lesson;
- the lesson year is within 1990–2009; the filter matches the lesson metadata;
- every relevant_lesson_id exists in the dataset.

Output: gold_set_hard.json (read by check_qdrant.py)
"""
import json
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

DATASET = "llis_full.json"
CHUNKS = "llis_chunks.jsonl"
OUT = "gold_set_hard"
YEARS = (1990, 2009)
SECTION_KEYS = {
    "Driving Event": "driving_event",
    "Lesson Learned": "lesson_learned",
    "Recommendation": "recommendation",
    "Evidence": "evidence",
    "References": "references",
}

gold_items = [
    # ---------- Materials, thermal, mechanisms ----------
    {
        "lesson_id": "535",
        "question": "Why would small ceramic plates glued to a metal block crack when the assembly is chilled down, even though the glue passed tests on flat coupons?",
        "ground_truth_answer": "Different thermal contraction produces not only shear in the bond but also bending stresses when the mounting lets parts push on each other.",
        "context_quote": "it can also produce bending stresses if the mounting design is not carefully planned for cryogenic temperatures.",
        "section": "Lesson Learned",
        "category": "Physical & Thermal Mechanics",
        "review_notes": "Paraphrased: no 'cryogenic', 'epoxy', 'CTE' in the question.",
    },
    {
        "lesson_id": "535",
        "question": "What assembly change finally stopped the substrate boards on the copper cube from breaking during cooldown?",
        "ground_truth_answer": "Mounting the boards so that they were not in contact with each other.",
        "context_quote": "When the MIDAS cube was assembled with the boards not in contact, so that no stresses would be produced during shrinkage, the boards did not shatter during cooldown.",
        "section": "Evidence",
        "category": "Corrective Action Evidence",
        "review_notes": "Evidence section: confirms the fix worked.",
    },
    {
        "lesson_id": "487",
        "question": "Can rubber or plastic parts make a thruster feed system misbehave from propellant fumes even when there is no chemical attack?",
        "ground_truth_answer": "Yes: non-metallic components can swell in propellant vapor and cause malfunction even without chemical incompatibility.",
        "context_quote": "Swelling of non-metallic components upon exposure to propellant vapors can result in propulsion system malfunction even if no chemical incompatibility exists.",
        "section": "Lesson Learned",
        "category": "Materials Compatibility",
        "review_notes": "Paraphrased ('rubber or plastic parts', 'fumes').",
    },
    {
        "lesson_id": "394",
        "question": "How do you get a believable lifetime estimate for a spacecraft's rotating actuator before signing off the design?",
        "ground_truth_answer": "Run life cycle testing under realistic conditions before final design acceptance and flight qualification.",
        "context_quote": "On mission critical rotating mechanical equipment, life cycle testing should be performed under realistic conditions to obtain a reasonably accurate life-time forecast prior to final design acceptance and flight qualification.",
        "section": "Recommendation",
        "category": "Test & Verification Practice",
        "review_notes": "Voyager scan platform lesson (1995); question does not name Voyager.",
    },
    {
        "lesson_id": "642",
        "question": "Why can't we assume a propellant tank made of supposedly nonmagnetic steel won't disturb the spacecraft's attitude?",
        "ground_truth_answer": "A component can have a significant magnetic moment despite non-magnetic constituent materials, e.g. magnetized during fabrication or handling.",
        "context_quote": "A spacecraft component may possess a significant magnetic moment despite the non-magnetic characteristics of the constituent materials.",
        "section": "Lesson Learned",
        "category": "Root Cause Analysis",
        "review_notes": "Paraphrased; no 'magnetized'/'stainless' wording from the lesson learned.",
    },
    {
        "lesson_id": "1040",
        "question": "Why is fixing Teflon creep out of shuttle thruster valve seats a design problem rather than a matter of choosing a better material?",
        "ground_truth_answer": "Plastics and metals differ in thermal expansion by an order of magnitude, so engineering approaches (seal entrapment, time at temperature) are needed instead of materials approaches.",
        "context_quote": "Since the coefficients of thermal expansion of plastics and metals differ by an order of magnitude, preventing unwanted seal extrusion entails implementation of appropriate engineering rather than materials approaches.",
        "section": "Recommendation",
        "category": "Physical & Thermal Mechanics",
        "review_notes": "WSTF lesson (2001); 'Teflon creep' paraphrases 'PTFE seal extrusion'.",
    },
    # ---------- Test & verification ----------
    {
        "lesson_id": "1196",
        "question": "If we have to deviate from testing the vehicle exactly as it will operate in flight, how should that deviation be handled?",
        "ground_truth_answer": "Assess it carefully, use alternate measures such as independent validation, put it in the risk management plan, get senior management concurrence and report it at reviews.",
        "context_quote": "Departures from this principle must be reflected in the project risk management plan, communicated to senior management for concurrence, and reported at reviews.",
        "section": "Recommendation",
        "category": "Test & Verification Practice",
        "review_notes": "Avoids the literal phrase 'test as you fly'.",
    },
    {
        "lesson_id": "771",
        "question": "On which JPL flight projects was stress-testing electronics beyond flight limits used instead of analysis?",
        "ground_truth_answer": "Magellan, NASA Scatterometer, Microwave Limb Sounder, Galileo and Cassini.",
        "context_quote": "This practice has been used on Magellan, NASA scatterometer, Microwave Limb Sounder, Galileo, Cassini",
        "section": "Evidence",
        "category": "Program Heritage",
        "review_notes": "Evidence section lists missions; question names none of them.",
    },
    {
        "lesson_id": "823",
        "question": "Why shake a payload at low levels and measure its response if we already have a finite element model?",
        "ground_truth_answer": "Modal testing verifies the analytical models (FEMs); without it model errors and uncertainty can lead to structural failure.",
        "context_quote": "Modal testing is used to verify analytical models or FEMs of a structure or payload.",
        "section": "Lesson Learned",
        "category": "Test & Verification Practice",
        "review_notes": "'shake ... measure response' paraphrases 'modal testing'.",
    },
    {
        "lesson_id": "806",
        "question": "What are the goals of running a heavily sensored hydrogen-oxygen engine as a technology testbed?",
        "ground_truth_answer": "Evaluate new propulsion technologies, validate or modify analytical models, understand engine operation under varying conditions and ensure reliability and operability.",
        "context_quote": "(2) validate or modify analytical models;",
        "section": "Recommendation",
        "category": "Test & Verification Practice",
        "review_notes": "MSFC SSME Technology Test Bed practice (1999).",
    },
    {
        "lesson_id": "1529",
        "question": "What analysis should be done on ground support gear so automated test software can't overdrive flight hardware?",
        "ground_truth_answer": "Perform an FMEA on the GSE test equipment to find failure modes that could make the test software command an overtest.",
        "context_quote": "perform a Failure Mode and Effects Analysis (FMEA) on GSE test equipment to identify failure modes that could cause test monitoring and control software to initiate an overtest.",
        "section": "Recommendation",
        "category": "Test & Verification Practice",
        "review_notes": "MRO multipaction test incident (2003).",
    },
    # ---------- Operations ----------
    {
        "lesson_id": "584",
        "question": "A Mars orbiter accidentally repeated a drag-pass trim burn; what configuration-control lesson came out of it?",
        "ground_truth_answer": "Spacecraft and sequence teams must be notified of ground software changes, and uplink sequences must be consistent with onboard memory partitions.",
        "context_quote": "In particular, uplink sequences must be consistent with memory partitions onboard the spacecraft.",
        "section": "Lesson Learned",
        "category": "Operations",
        "review_notes": "MGS aerobraking anomaly; 'aerobraking maneuver' paraphrased.",
    },
    {
        "lesson_id": "1521",
        "question": "Which JPL procedure document governs how command loads are radiated to spacecraft?",
        "ground_truth_answer": "Operation of Uplink Process, Rev. 2, JPL Doc ID 27232 (July 19, 2002).",
        "context_quote": "Operation of Uplink Process, Rev. 2, JPL Doc ID 27232, July 19, 2002.",
        "section": "References",
        "category": "Standards & References",
        "review_notes": "References section.",
    },
    {
        "lesson_id": "1531",
        "question": "Right after launch the ground station reports a weaker downlink than predicted; should we treat it as a real degradation?",
        "ground_truth_answer": "Not necessarily: uplink feedthrough and saturation cause apparent degradation, and reported levels approach predictions as the spacecraft moves away.",
        "context_quote": "As the spacecraft moves farther from Earth, the reported signal levels tend to move closer to predicted values.",
        "section": "Lesson Learned",
        "category": "Operations",
        "review_notes": "No 'feedthrough' or 'saturation' in the question.",
    },
    {
        "lesson_id": "1480",
        "question": "Why keep landing-sequence settings changeable until a few hours before arrival at a planet?",
        "ground_truth_answer": "Being able to alter critical flight-software parameters within hours of critical events in response to unexpected data can save the mission.",
        "context_quote": "Critical parameters coded in flight software and the ability to alter them within hours of critical events in response to unexpected data on flight characteristics can save a planetary mission or deep space encounter.",
        "section": "Lesson Learned",
        "category": "Operations",
        "review_notes": "MER EDL parameter update lesson; 'EDL' not used.",
    },
    # ---------- Safety & ground operations ----------
    {
        "lesson_id": "2196",
        "question": "Do toxic storable-propellant mishaps tend to happen while the system sits idle or while people are working on it?",
        "ground_truth_answer": "Most incidents happen during some sort of dynamic operation; the propellants are stable in a static configuration.",
        "context_quote": "One can deduce, however, that most incidents happen during some sort of dynamic operation.",
        "section": "Recommendation",
        "category": "Safety",
        "review_notes": "Filtered query (center).",
        "filter": {"org_code": "KSC"},
    },
    {
        "lesson_id": "2056",
        "question": "A contractor fell off a building during panel replacement; what was wrong with how the job was classified?",
        "ground_truth_answer": "It was treated as low-slope roofing but should have been classified as steel erection, which requires physical fall restraints.",
        "context_quote": "The roofing work should have been classified as a steel erection, for which the Occupational Health and Safety Administration (OSHA) required physical restraints.",
        "section": "Lesson Learned",
        "category": "Safety",
        "review_notes": "KSC Building M6-794 mishap; no OSHA wording in the question.",
    },
    {
        "lesson_id": "557",
        "question": "At what point should the batteries in an uninterruptible supply be replaced?",
        "ground_truth_answer": "When about 50% capability is reached (low level indicator).",
        "context_quote": "suggested guideline is when 50% capability is reached",
        "section": "Recommendation",
        "category": "Quantitative Engineering Values",
        "review_notes": "Filtered query (center).",
        "filter": {"org_code": "JSC"},
    },
    {
        "lesson_id": "406",
        "question": "What automatic protection was added to the hydrogen exhaust piping on the engine test stand to catch air leaking in?",
        "ground_truth_answer": "An O2 meter that, at about 0.5% oxygen, sounds an audible alarm and starts an automatic gaseous nitrogen purge.",
        "context_quote": "the meter both produces an audible alarm and an automatic gasseous nitrogen purge is initiated.",
        "section": "Evidence",
        "category": "Corrective Action Evidence",
        "review_notes": "Filtered query (center). Stennis B-1 vent line rupture (1996).",
        "filter": {"org_code": "SSC"},
    },
    {
        "lesson_id": "1338",
        "question": "After a worker was pinned by rolling support equipment on an elevated deck, what rule was added about moving equipment?",
        "ground_truth_answer": "Always verify equipment movement visually and remove visual obstructions before moving mobile equipment.",
        "context_quote": "Always verify movement of equipment visually.",
        "section": "Recommendation",
        "category": "Safety",
        "review_notes": "Stennis A1 test stand near-miss (2002).",
    },
    {
        "lesson_id": "1622",
        "question": "Which Dryden procedure chapter applies to the incident where a pressurized nitrogen servicing line struck an aircraft?",
        "ground_truth_answer": "DCP-S-009, Chapter 8.",
        "context_quote": "DCP - S - 009, Chapter 8",
        "section": "References",
        "category": "Standards & References",
        "review_notes": "References section; 'hose', 'GN2' paraphrased.",
    },
    {
        "lesson_id": "1095",
        "question": "What should submittal reviewers check so that ordered building HVAC units actually match the available electrical supply?",
        "ground_truth_answer": "Understand what every model code number and letter means and verify utility service connection requirements for major equipment.",
        "context_quote": "Make sure that the reviewer understands what every one of those specific model code numbers and letters means.",
        "section": "Recommendation",
        "category": "Facilities",
        "review_notes": "Filtered query (center). 480 V vs 208 V HVAC mismatch.",
        "filter": {"org_code": "AFRC"},
    },
    {
        "lesson_id": "746",
        "question": "During office renovation projects, who should handle relocating the occupants so the project manager isn't overloaded?",
        "ground_truth_answer": "A dedicated Move Coordinator role within the project team.",
        "context_quote": "Create the role of a Move Coordinator within the Project Team.",
        "section": "Recommendation",
        "category": "Facilities",
        "review_notes": "Filtered query (center).",
        "filter": {"org_code": "GRC"},
    },
    # ---------- Quality, manufacturing, NDE ----------
    {
        "lesson_id": "463",
        "question": "How did rework on a booster engine regulator end up putting a satellite into the wrong orbit?",
        "ground_truth_answer": "A reworked set screw rotated and decayed regulator output because rework instructions lacked torquing and QA verification requirements.",
        "context_quote": "Reworked screws can be the cause of decayed precision regulator output if rework instructions lack requirements for set screw torquing and verification by a quality inspector.",
        "section": "Lesson Learned",
        "category": "Root Cause Analysis",
        "review_notes": "Atlas/Centaur AC-74 (1993).",
    },
    {
        "lesson_id": "733",
        "question": "At Goddard, where do anomalies seen in orbit get written up, as opposed to failures during ground testing?",
        "ground_truth_answer": "In a separate Spacecraft Orbital Anomaly Report (SOAR) system.",
        "context_quote": "A separate system, the \"Spacecraft Orbital Anomaly Report (SOAR)\", is used for the reporting, evaluation and correction of problems occurring on-orbit",
        "section": "Recommendation",
        "category": "Quality Assurance",
        "review_notes": "Filtered query (center).",
        "filter": {"org_code": "GSFC"},
    },
    {
        "lesson_id": "801",
        "question": "On which propulsion programs has 3D X-ray CT imaging been used to inspect hardware?",
        "ground_truth_answer": "Solid Rocket Motor, SSME, Solid Propulsion Integrity Program, Modified NASA Motor Program and a sub-scale solid rocket combustion simulator.",
        "context_quote": "This practice has been used on Solid Rocket Motor (SRM), Space Shuttle Main Engine (SSME), Solid Propulsion Integrity Program (SPIP), Modified NASA Motor Program, Sub-Scale Solid Rocket Combustion Simulator",
        "section": "Evidence",
        "category": "Program Heritage",
        "review_notes": "Filtered query (center). Evidence section.",
        "filter": {"org_code": "MSFC"},
    },
    {
        "lesson_id": "1111",
        "question": "Without taking a mechanism apart, how can you tell whether its explosive actuator actually fired?",
        "ground_truth_answer": "Measure the firing circuit resistance before and after firing.",
        "context_quote": "Currently, the only test being performed to verify that the initiator fired, without disassembly of the pyrotechnic mechanism, is to measure firing circuit resistance before and after firing.",
        "section": "Evidence",
        "category": "Test & Verification Practice",
        "review_notes": "ASAP ISS EVA lesson; NASA response in Evidence.",
    },
    # ---------- Program management ----------
    {
        "lesson_id": "921",
        "question": "When a contractor writes mission-critical flight code, which three areas must the center's acquisition plan cover?",
        "ground_truth_answer": "Software requirements, software management planning and acceptance testing.",
        "context_quote": "These plans must provide for Software Requirements, Software Management Planning, and Acceptance Testing.",
        "section": "Recommendation",
        "category": "Software Management",
        "review_notes": "MCO-era software acquisition lesson.",
    },
    {
        "lesson_id": "1405",
        "question": "Why bring engineers from outside a project to attack a stubborn technical problem?",
        "ground_truth_answer": "A 'fresh look' from outside experts uncovers issues the original engineering team missed.",
        "context_quote": "This “fresh look” approach is valuable in uncovering issues not found by the original engineering team.",
        "section": "Lesson Learned",
        "category": "Organization",
        "review_notes": "NESC tiger-team lesson; 'tiger team' and 'One NASA' avoided.",
    },
    {
        "lesson_id": "1063",
        "question": "How did the Station program respond to the safety panel's concern that the alarm system lacked systems engineering?",
        "ground_truth_answer": "It created a C&W Systems Integration Team (CWSIT) to manage all caution and warning development.",
        "context_quote": "The ISS Program concurs with these concerns and has instituted a C&W Systems Integration Team (CWSIT) to provide direction and management of all C&W development activities",
        "section": "Evidence",
        "category": "Corrective Action Evidence",
        "review_notes": "ASAP 1998; answer in NASA response (Evidence).",
    },
    {
        "lesson_id": "1113",
        "question": "How many spacesuit backpack life support units did NASA plan to hold after buying the extra one recommended by the safety panel?",
        "ground_truth_answer": "16 LSS units.",
        "context_quote": "NASA also plans to go forward with the recommendation to procure an additional LSS to achieve 16 LSS's.",
        "section": "Evidence",
        "category": "Quantitative Engineering Values",
        "review_notes": "Number inside a long Evidence section.",
    },
    {
        "lesson_id": "1047",
        "question": "How many government-required quality checkpoints in orbiter processing were expected to remain after the reduction effort?",
        "ground_truth_answer": "About 6,000 of the original 22,000 GMIPs.",
        "context_quote": "It is estimated that approximately 6,000 of the original 22,000 GMIPs will remain in place at the end of this effort.",
        "section": "Evidence",
        "category": "Quantitative Engineering Values",
        "review_notes": "'GMIP' not used in the question.",
    },
    {
        "lesson_id": "1137",
        "question": "What software tool was built to let shuttle engineers edit work paperwork more quickly?",
        "ground_truth_answer": "WAVE (work authorization document authoring and validation environment).",
        "context_quote": "development of the new work authorization document authoring and validation environment (WAVE) computer software to allow engineers to quickly and easily modify work instructions",
        "section": "Evidence",
        "category": "Corrective Action Evidence",
        "review_notes": "Evidence section.",
    },
    {
        "lesson_id": "1792",
        "question": "Which Ames procedural document governs how new-business proposals are authorized and reviewed?",
        "ground_truth_answer": "APR 8000.1, Procedure for Proposal Authorization, Development, Review, Approval and Archiving.",
        "context_quote": "Procedure for Proposal Authorization, Development, Review, Approval and Archiving, APR 8000.1",
        "section": "References",
        "category": "Standards & References",
        "review_notes": "Filtered query (center). References section.",
        "filter": {"org_code": "ARC"},
    },
    {
        "lesson_id": "1289",
        "question": "For a small classified team where each specialist is irreplaceable, what should the staffing plan include?",
        "ground_truth_answer": "Cross training, mentoring, a pre-identified pool of cleared specialists, risk-based schedules and an adequate overtime budget.",
        "context_quote": "pre-identified pool of specialists with appropriate security backgrounds",
        "section": "Recommendation",
        "category": "Project Management",
        "review_notes": "'one deep' wording avoided.",
    },
    {
        "lesson_id": "1476",
        "question": "Why did the Columbia board's web evidence database end up barely used by investigators?",
        "ground_truth_answer": "Investigators lacked time to browse links on a PC and the tool could not produce logic tree reports.",
        "context_quote": "A combination of factors, including lack of time by investigators to search through the IO links on a personal computer and IO's lack of logic tree report capability, severely limited use of the system by other than personnel keying in IO data.",
        "section": "Lesson Learned",
        "category": "Accident Investigation",
        "review_notes": "'Investigation Organizer' not named.",
    },
    {
        "lesson_id": "1213",
        "question": "A late tweak to an EVA hardware cover confused the spacewalkers because the checklist wasn't updated; what is the process takeaway?",
        "ground_truth_answer": "Configuration control must cover all levels of authorized change and feed back to affected elements, including training and operations.",
        "context_quote": "Configuration control processes must account for all levels of authorized changes and provide feedback to affected program elements including training and operations.",
        "section": "Lesson Learned",
        "category": "Configuration Management",
        "review_notes": "STS-88 trunnion covers.",
    },
    {
        "lesson_id": "1479",
        "question": "Which NASA policy on limiting space junk did the end-of-life experience of the small nitric-oxide satellite suggest clarifying?",
        "ground_truth_answer": "NPD 8710.3A, NASA Policy for Limiting Orbital Debris Generation.",
        "context_quote": "NPD 8710.3A, NASA Policy for Limiting Orbital Debris Generation",
        "section": "References",
        "category": "Standards & References",
        "review_notes": "SNOE reentry; References section.",
    },
    {
        "lesson_id": "1490",
        "question": "Which ASTM method for counting airborne particles was cited for cleanroom work with foreign partners on the ozone instrument?",
        "ground_truth_answer": "ASTM-F25, Standard Test Method for Sizing and Counting Airborne Particulate Contamination in Clean Rooms.",
        "context_quote": "ASTM-F25 Standard Test Method for Sizing and Counting Airborne Particulate Contamination in Clean Rooms and Other Dust-Controlled Areas",
        "section": "References",
        "category": "Standards & References",
        "review_notes": "SAGE III; References section.",
    },
    # ---------- Program-filtered questions (2000s lessons with program metadata) ----------
    {
        "lesson_id": "2216",
        "question": "What should institutions do so routine building services don't interfere with a project's lockdown during critical operations?",
        "ground_truth_answer": "Set up institutional procedures coordinating nonessential infrastructure services with configuration control freezes requested by flight projects.",
        "context_quote": "Establish and implement institutional procedures for coordination of nonessential infrastructure services with scheduled configuration control freezes that have been instituted at the request of flight projects during critical mission operations.",
        "section": "Recommendation",
        "category": "Operations",
        "review_notes": "Filtered query (program).",
        "filter": {"program": "Mars Phoenix"},
    },
    {
        "lesson_id": "1996",
        "question": "Even with guide rails, how should handlers steer a wheeled fixture carrying flight hardware?",
        "ground_truth_answer": "Always attach a steering stick or T-bar to the casters or wheels.",
        "context_quote": "Even when rails or tracks are used to constrain motion, always attach a steering stick or T-bar to the casters or wheels to guide the movement of valuable equipment.",
        "section": "Recommendation",
        "category": "Ground Handling",
        "review_notes": "Filtered query (program). Heatshield handling incident.",
        "filter": {"program": "Mars Science Laboratory"},
    },
    {
        "lesson_id": "1864",
        "question": "How do you keep a tube amplifier's power supply noise off the spacecraft bus?",
        "ground_truth_answer": "Add an external filter between the TWTA and the spacecraft bus.",
        "context_quote": "Missions using TWTAs and needing low levels of EMI should incorporate an external filterbetween the TWTA and the spacecraft bus.",
        "section": "Recommendation",
        "category": "Design Practice",
        "review_notes": "Filtered query (program). Note: other lessons on the same TWTA use different program spellings.",
        "filter": {"program": "Lunar Reconnaissance Orbiter"},
    },
    {
        "lesson_id": "1781",
        "question": "How do you make sure testbed backups can actually be restored?",
        "ground_truth_answer": "Have a system administrator familiar with both the script and the backup hardware verify the backups' actual performance.",
        "context_quote": "Assure that the actual performance of critical backups is verified by a computer system administrator familiar with both the script and the backup hardware.",
        "section": "Recommendation",
        "category": "Operations",
        "review_notes": "Filtered query (program).",
        "filter": {"program": "Cassini"},
    },
    {
        "lesson_id": "1725",
        "question": "What should test plans assume when firing cable cutters near sensitive flight hardware?",
        "ground_truth_answer": "Assume hazardous blow-by will occur: protect sensitive areas from venting and contamination, use barriers, fire remotely.",
        "context_quote": "Always consider pyro venting a possibility and protect sensitive areas of the flight hardware against contamination.",
        "section": "Recommendation",
        "category": "Safety",
        "review_notes": "Filtered query (program).",
        "filter": {"program": "Mars Exploration Rover"},
    },
    {
        "lesson_id": "2037",
        "question": "How did expensive station spare circuit boards end up being sold as scrap?",
        "ground_truth_answer": "Usage rate was the only criterion for excessing; the holding fixtures had no usage history, so the attached flight boards went with them.",
        "context_quote": "The sole decision criterion for excessing items was usage rate.",
        "section": "Lesson Learned",
        "category": "Logistics",
        "review_notes": "Filtered query (program).",
        "filter": {"program": "ISS program"},
    },
    # ---------- Multi-lesson questions (any relevant lesson counts as a hit) ----------
    {
        "lesson_id": "641",
        "question": "What units mix-up caused the loss of the 1999 Mars weather orbiter?",
        "ground_truth_answer": "Thruster small-forces data in a ground software file was in English units (pound-force seconds) instead of the required metric units.",
        "context_quote": "the root cause for the loss of the MCO spacecraft was the failure to use metric units in the coding of a ground software file",
        "section": "Lesson Learned",
        "category": "Root Cause Analysis",
        "review_notes": "Multi-lesson: MIB report and software development lesson.",
        "relevant_lesson_ids": ["641", "740"],
    },
    {
        "lesson_id": "938",
        "question": "Why did the descent engines of the 1999 Mars south-pole lander probably shut off too early?",
        "ground_truth_answer": "Spurious leg-deployment signals were taken as valid touchdown, shutting down the engines at about 40 m.",
        "context_quote": "It is believed that these momentary signals were recorded as valid touchdown events, resulting in the engines shutting down at an altitude of 40 meters.",
        "section": "Driving Event",
        "category": "Root Cause Analysis",
        "review_notes": "Multi-lesson: probable scenario lesson and test-as-you-fly lesson.",
        "relevant_lesson_ids": ["938", "1196"],
    },
    {
        "lesson_id": "331",
        "question": "What did the Hubble mirror flaw teach about the expertise and independence of quality oversight staff?",
        "ground_truth_answer": "QA needs technical expertise in specialised areas such as optics and must be independent of program and engineering management.",
        "context_quote": "If quality assurance personnel responsible for oversight of the quality of highly technical state-of-the-art development do not have a degree of expertise in that technical area, the likelihood of discovering QA problems decreases significantly.",
        "section": "Lesson Learned",
        "category": "Quality Assurance",
        "review_notes": "Multi-lesson: twin MSFC 1993 lessons (expertise / access).",
        "relevant_lesson_ids": ["331", "332"],
    },
    {
        "lesson_id": "1196",
        "question": "Why should ground tests reproduce the real mission profile and flight configuration?",
        "ground_truth_answer": "End-to-end verification may be compromised when it is not consistent with the mission profile plus margin and off-design parameters.",
        "context_quote": "The process of end-to-end system verification (either through testing, simulation, or analysis) may be compromised when it is not consistent with the mission profile (plus margin and the appropriate off-design parameters).",
        "section": "Lesson Learned",
        "category": "Test & Verification Practice",
        "review_notes": "Multi-lesson: all 1990–2009 lessons that invoke the test-as-you-fly principle.",
        "relevant_lesson_ids": ["1196", "1335", "909", "1733", "1390", "1713", "1416", "2044", "2336", "2116"],
    },
    {
        "lesson_id": "987",
        "question": "What do older program post-mortems say about letting the list of system requirements keep growing during development?",
        "ground_truth_answer": "Keep requirements stable and do not allow requirements creep; involve all stakeholders when finalizing them.",
        "context_quote": "Stable Requirements are key to program stability. Do not allow requirements to creep.",
        "section": "Lesson Learned",
        "category": "Project Management",
        "review_notes": "Multi-lesson: Chandra, F-18 HARV, X-34 FASTRAC, OSP and other lessons on requirements creep.",
        "relevant_lesson_ids": ["987", "1604", "17802", "1420", "1522", "1506", "1956"],
    },
]

STOPWORDS = set("""a an the of to in on for and or is are was were be been being what which who whom why how when where
does do did can could should would will with without from by as at that this these those it its into than then so
if not no any all each every our we you your they their there about after before during even still actually such
just more most less very much many have has had get got make made end up out off""".split())


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def lexical_overlap(question: str, quote: str) -> float:
    """Share of the question's content words that literally occur in the quote (BM25 'leak')."""
    q = content_words(question)
    return len(q & content_words(quote)) / len(q) if q else 0.0


def main():
    with open(DATASET, encoding="utf-8") as f:
        by_id = {r["id"]: r for r in json.load(f)}
    with open(CHUNKS, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    errors, records = [], []
    for n, item in enumerate(gold_items, 1):
        item = {"id": n, **item}
        lid, quote = item["lesson_id"], item["context_quote"]
        doc = by_id.get(lid)
        if doc is None:
            errors.append(f"Q{n}: lesson {lid} not found")
            continue
        if not YEARS[0] <= (doc["year"] or 0) <= YEARS[1]:
            errors.append(f"Q{n}: lesson {lid} year {doc['year']} outside {YEARS}")
        if quote not in doc["full_text"]:
            errors.append(f"Q{n}: quote not in full_text of {lid}")
        if quote not in doc["sections"].get(SECTION_KEYS[item["section"]], ""):
            errors.append(f"Q{n}: quote not in section {item['section']} of {lid}")
        hit_chunks = [c["id"] for c in chunks if lid in c["metadata"]["lesson_ids"] and quote in c["page_content"]]
        if not hit_chunks:
            errors.append(f"Q{n}: quote not inside a single chunk of lesson {lid}")
        for field, value in item.get("filter", {}).items():
            if doc.get(field) != value:
                errors.append(f"Q{n}: filter {field}={value!r} does not match lesson value {doc.get(field)!r}")
        rel = item.setdefault("relevant_lesson_ids", [lid])
        if lid not in rel:
            errors.append(f"Q{n}: relevant_lesson_ids must include {lid}")
        errors += [f"Q{n}: relevant lesson {r} not found" for r in rel if r not in by_id]

        item["quote_chunk_ids"] = hit_chunks
        item["lexical_overlap"] = round(lexical_overlap(item["question"], quote), 2)
        item["document_title"] = doc["title"]
        item["nasa_center"] = doc["org_code"] or "NASA"
        item["url"] = doc["url"]
        item["date"] = doc["date"]
        item["year"] = doc["year"]
        records.append(item)

    if errors:
        print("VERIFICATION FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)

    print(f"Verified {len(records)} records against {DATASET} and {CHUNKS}.")
    print("Sections  :", dict(Counter(r["section"] for r in records)))
    print("Centers   :", dict(Counter(r["nasa_center"] for r in records)))
    print("Decades   :", dict(Counter(f"{r['year'] // 10 * 10}s" for r in records)))
    print("Filtered  :", sum("filter" in r for r in records),
          "| multi-lesson:", sum(len(r["relevant_lesson_ids"]) > 1 for r in records))
    overlaps = [r["lexical_overlap"] for r in records]
    print(f"Lexical overlap question->quote: mean {sum(overlaps) / len(overlaps):.2f}, max {max(overlaps):.2f}")

    # JSON for check_qdrant.py / Ragas
    with open(f"{OUT}.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {OUT}.json")


if __name__ == "__main__":
    main()
