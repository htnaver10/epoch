// The numbers the planner and the simulation rest on, with where each came from. Where the
// sources give no figure the value is a modelling assumption and is marked as one.

/** Head fire intensity (kW/m) up to which each means of attack works. */
export const LIMIT = {
  handLine: 500, // class 2: "constructed fireguard should hold" (Alexander & De Groot 1988)
  crew: 2000, // class 3: ground crews with pumps and tools; above it they need air support (Wotton et al. 2017)
  machineGuard: 2000, // a 6 m dozer guard stops surface fire only
  bucket: 2000, // helicopter bucket: light loads have under a 5% chance on a rank 4 edge (Wheatley & Wotton 2026), so buckets take class 3 and below
  skimmer: 10000, // heavy skimmers: one load is often enough at rank 2, about five stacked at rank 4, eight or more at rank 5, and 10,000 kW/m is the realistic ceiling (same study)
  retardant: 4000, // same threshold for a retardant line
  coatedGuard: 6000, // assumption: a machine guard coated with retardant, as cut west of Yellowknife in 2023
  any: 10000, // class 6: no direct action at all
};

/** Line production, km per hour (NWCG Fireline Production Rates, timber fuel models 8-10). */
export const RATE = {
  crewDirect: 0.211, // Type 1 20-person crew, direct attack and mop-up
  crewIndirect: 0.139, // the same crew cutting indirect line
  initialAttack: 0.32, // 4-person crew in its first hours on a new fire: 16 chains an hour (NWCG initial action table)
  dozer: 0.55, // D7/D8 class in timber with understory: 400-700 m/h on gentle slopes
  sprinklerSetup: 0.5, // assumption: a structure protection unit lays 500 m of sprinkler line an hour
};

export const DROP = {
  heavyLineKm: 0.32, // a 3,000 US gal load lays 279 m at coverage level 4, short-needle conifer with heavy litter (NWCG PMS 505, table 9): two 160 m cells
  litresPerKmRetardant: 31000,
  knockdownBase: 6000, // litres to knock down 160 m of quiet edge: one CL-415 load, which is about that long on the ground
  knockdownPerKw: 8, // and more for a hotter one: 30,000 L (five loads) at 3,000 kW/m, fitted to Wheatley & Wotton (2026)
  wetLitres: 6000, // litres to wet a 160 m cell of unburned fuel: one heavy skimmer load
  aheadMinMinutes: 4, // water goes down no closer than this ahead of the fire: the aircraft has to get out
  aheadShare: 0.7, // a drop ahead of the fire slows it but does not stop it: what it is worth against a knockdown. Assumption.
  wetMinutes: 25, // water and foam work for minutes: a drop's effect in the stand is gone in 25 to 76 min (Wheatley et al. 2023), so it goes down right ahead of the fire
  knockMinutes: 60, // how long a knocked-down edge stays down without a crew. Assumption.
};

export const SOURCES = [
  ["Aerial tactics", "NWCG Standards for Aerial Supervision, PMS 505: direct, parallel and indirect attack; retardant outside the perimeter; anchor, flank and pinch", "https://www.nwcg.gov/publications/pms505"],
  ["Skimmer drops", "Wheatley & Wotton (2026), Int. J. Wildland Fire 35: CL-415 drops needed by intensity rank, from 1,803 Ontario fires", "https://connectsci.au/wf/article/doi/10.1071/WF25311"],
  ["Water on fuel", "Wheatley et al. (2023), Int. J. Wildland Fire 32: how long a water drop changes conditions in a jack pine stand", "https://connectsci.au/wf/article/doi/10.1071/WF22218"],
  ["Aircraft effectiveness", "USDA Forest Service, Aerial Firefighting Use and Effectiveness report (2020)", "https://www.fs.usda.gov/sites/default/files/2020-08/08192020_afue_final_report_0.pdf"],
  ["Where to treat", "Finney (2007), treating the major fire travel routes on a minimum-travel-time spread; Pais et al. (2021), downstream protection value", "https://connectsci.au/wf/article/16/6/702/23005/A-computational-method-for-optimising-fuel"],
  ["Fire behaviour", "Canadian Forest Fire Behavior Prediction System (Forestry Canada 1992, ST-X-3; cffdrs_r reference code)", "https://github.com/cffdrs/cffdrs_r"],
  ["Intensity classes", "Fire intensity class table, after Alexander & De Groot (1988)", "https://wfm.gov.sk.ca/static/safety/21-fire_intensity.pdf"],
  ["Suppression limits", "Wotton, Flannigan & Marshall (2017), Environ. Res. Lett. 12", "https://iopscience.iop.org/article/10.1088/1748-9326/aa7e6e"],
  ["Tactics by rank", "BC Wildfire Service, wildfire rank", "https://www2.gov.bc.ca/gov/content/safety/wildfire-status/wildfire-response/about-wildfire/wildfire-rank"],
  ["Line production", "NWCG Fireline Production Rates (USDA Forest Service T&D)", "https://www.fs.usda.gov/t-d/nwcg/"],
  ["Dozer guards", "Dozer Boss, Firefighting in Canada", "https://www.firefightingincanada.com/wildland/dozer-boss-25658"],
  ["Retardant coverage", "USDA Forest Service coverage level table", "https://www.fs.usda.gov/rm/fire/pubs/htmlpubs/htm00572834/page01.htm"],
  ["Aircraft", "GNWT air tanker fleet; Conair, De Havilland and Air Spray specifications; Alberta rotor-wing sheet", "https://www.gov.nt.ca/ecc/en/services/wildfire-operations/air-tanker-fleet"],
  ["Skimmer drops", "Nova Scotia DNR, water drop safety", "https://novascotia.ca/natr/forestprotection/wildfire/bffsc/lessons/lesson6/waterdrop.asp"],
  ["Structure protection", "BC structure protection units; FireSmart home ignition zone", "https://firesmartbc.ca/the-home-ignition-zone/"],
  ["Yellowknife 2023", "Cabin Radio, inside the NWT's 2023 wildfire decision-making", "https://cabinradio.ca/172113/news/yellowknife/inside-the-nwts-2023-wildfire-decision-making/"],
  ["Jasper 2024", "Parks Canada, Jasper wildfire timeline", "https://parks.canada.ca/pn-np/ab/jasper/visit/feu-alert-fire/feudeforet-jasper-wildfire/chronologie-timeline"],
  ["Priorities", "Alberta Wildfire: life, communities, then environmental and economic values", "https://www.alberta.ca/how-we-fight-wildfires"],
];
