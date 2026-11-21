/**
 * FloraGuard AI - Plant Disease Knowledge Base & Pathogen Dataset
 * Comprehensive botanical pathology references, symptoms, risk factors, and IPM treatment plans.
 */

export const DISEASE_DATABASE = {
  tomato_early_blight: {
    diseaseKey: 'tomato_early_blight',
    canonicalClass: 'tomato__early_blight',
    plant: 'Tomato',
    disease: 'Early Blight',
    pathogen: 'Alternaria solani (Ellis & G. Martin)',
    category: 'Fungal Ascomycota',
    severity: 'Moderate',
    severityScore: 62,
    confidence: 0.982,
    probabilities: {
      'Tomato — Early Blight': 0.982,
      'Tomato — Late Blight': 0.012,
      'Tomato — Septoria Leaf Spot': 0.004,
      'Tomato — Healthy': 0.002,
    },
    description:
      'Early blight is a major foliar fungal disease of solanaceous crops. It begins as small, dark brown-to-black necrotic spots on the oldest lower foliage, developing characteristic concentric ring patterns resembling an archery target, often surrounded by a chlorotic yellow halo.',
    cause:
      'Soil-borne and crop-residue-borne fungal pathogen Alternaria solani producing alternariol toxins.',
    transmission:
      'Conidia spores overwinter in solanaceous plant debris and weed hosts, spreading through wind-driven rain, sprinkler splash, and gardening shears.',
    favorableConditions:
      'Warm temperatures (24°C - 29°C / 75°F - 85°F), extended leaf wetness (>8 hours), and intermittent rain or high relative humidity (>80%).',
    symptoms: [
      'Concentric circular target-board spots on lower and older leaves',
      'Expanding yellow chlorosis surrounding dark necrotic lesions',
      'Upward curling, collar rot on lower stems, and premature leaf drop',
      'Defoliation exposing maturing fruit to physiological sunscald',
      'Dark sunken leathery lesions at the fruit stem calyx attachment',
    ],
    riskFactors: [
      'Overhead sprinkler irrigation wetting canopy foliage',
      'Crowded planting density restricting canopy ventilation',
      'Uncollected infected solanaceous crop debris from previous season',
      'Nutrient stress (especially low nitrogen / potassium ratio)',
    ],
    immediateActions: [
      'Sanitize pruning shears in 70% isopropyl alcohol and prune off all blighted lower leaves up to 18 inches off the soil.',
      'Transition immediately from overhead spraying to root-zone drip irrigation.',
      'Apply a 2-3 inch organic straw mulch layer to stop soil-borne fungal spore splashback onto leaves.',
      'Treat remaining foliage with registered copper hydroxide or biofungicidal Bacillus subtilis spray.',
    ],
    preventiveActions: [
      'Practice a strict 3-4 year solanaceous crop rotation (avoid planting tomato, potato, or eggplant in the same soil).',
      'Select certified disease-tolerant cultivars with high field resistance.',
      'Space plants 24-36 inches apart with vertical trellising for optimal airflow.',
      'Sterilize all garden tools, stakes, and tomato cages at the end of each season.',
    ],
    cropCarePlan: [
      { action: 'Sanitize & prune infected lower foliage', priority: 'High', time: 'Immediately (Day 1)' },
      { action: 'Install root-zone drip line & straw mulch', priority: 'High', time: 'Within 24 Hours' },
      { action: 'Foliar copper / Bacillus bio-protectant spray', priority: 'High', time: 'Within 48 Hours' },
      { action: 'Scout adjacent plants for concentric rings', priority: 'Medium', time: 'Every 2 Days' },
      { action: 'Maintain post-harvest soil sanitation', priority: 'Medium', time: 'Seasonal' },
    ],
  },

  corn_common_rust: {
    diseaseKey: 'corn_common_rust',
    canonicalClass: 'corn__common_rust',
    plant: 'Corn (Maize)',
    disease: 'Common Rust',
    pathogen: 'Puccinia sorghi (Schwein.)',
    category: 'Fungal Basidiomycota',
    severity: 'Moderate',
    severityScore: 58,
    confidence: 0.978,
    probabilities: {
      'Corn — Common Rust': 0.978,
      'Corn — Gray Leaf Spot': 0.014,
      'Corn — Northern Leaf Blight': 0.005,
      'Corn — Healthy': 0.003,
    },
    description:
      'Common corn rust produces prominent, elongated cinnamon-brown to brick-red pustules (uredinia) scattered across both upper and lower leaf surfaces. In advanced infections, the epidermal layer ruptures, releasing millions of powdery rust spores into the wind.',
    cause:
      'Airborne fungal pathogen Puccinia sorghi, which cannot overwinter in freezing climates and blows northward on storm systems.',
    transmission:
      'Long-distance wind dissemination of urediniospores from southern growing regions, activated by cool dew cycles.',
    favorableConditions:
      'Moderate daytime temperatures (16°C - 25°C / 60°F - 77°F), continuous high relative humidity (>95%), and 6 or more hours of morning dew.',
    symptoms: [
      'Small, elongated cinnamon-brown pustules erupting through leaf surfaces',
      'Powdery reddish-brown spore dust coating hands upon touching foliage',
      'Chlorotic yellow halos encircling densely clustered pustules',
      'Premature leaf senescence, rapid drying, and reduced photosynthetic capacity',
      'Stalk lodging and poor ear kernel fill under severe pre-silking pressure',
    ],
    riskFactors: [
      'Late-planted fields tasseling during peak rust spore arrival',
      'Planting susceptible sweet corn or non-Rp hybrid varieties',
      'Low canopy airflow and persistent river-valley fog or dew',
    ],
    immediateActions: [
      'Scout top ear leaves: if pustules cover >5% of leaves prior to tassel/silking, schedule a foliar fungicide application.',
      'Avoid high afternoon humidity by irrigating only during sunrise hours.',
      'Consult regional agronomy thresholds before applying strobilurin or triazole fungicides.',
    ],
    preventiveActions: [
      'Plant corn hybrids featuring Rp-gene resistance (Rp1-D, Rp1-E, or multigenic tolerance).',
      'Plant early in the season to ensure grain fill completes before peak spore influx.',
      'Maintain balanced soil potassium to enhance plant epidermal cell wall strength.',
    ],
    cropCarePlan: [
      { action: 'Evaluate pustule density on upper canopy leaves', priority: 'High', time: 'Immediately' },
      { action: 'Consult extension thresholds for fungicide timing', priority: 'High', time: 'Within 24 Hours' },
      { action: 'Monitor silking progress & ear development', priority: 'Medium', time: 'Weekly' },
      { action: 'Select resistant hybrid cultivars for next season', priority: 'Medium', time: 'Next Season' },
    ],
  },

  potato_late_blight: {
    diseaseKey: 'potato_late_blight',
    canonicalClass: 'potato__late_blight',
    plant: 'Potato',
    disease: 'Late Blight',
    pathogen: 'Phytophthora infestans (Mont.) de Bary',
    category: 'Oomycete (Water Mold)',
    severity: 'High',
    severityScore: 88,
    confidence: 0.985,
    probabilities: {
      'Potato — Late Blight': 0.985,
      'Potato — Early Blight': 0.011,
      'Potato — Healthy': 0.004,
    },
    description:
      'Late blight is historically the most devastating disease of potato and tomato, responsible for the 1840s Irish Potato Famine. It causes rapid necrosis, water-soaked dark lesions, white sporangial mold on leaf undersides, and destructive rot in tubers.',
    cause:
      'Aggressive oomycete water mold Phytophthora infestans that reproduces both asexually and sexually with extreme speed.',
    transmission:
      'Airborne sporangia travelling miles on moist winds, rain splash, and infected seed tubers.',
    favorableConditions:
      'Cool, wet, cloudy weather (15°C - 20°C / 59°F - 68°F) with relative humidity near 100% and frequent rainfall or heavy fog.',
    symptoms: [
      'Large, irregular dark water-soaked lesions expanding rapidly from leaf margins',
      'Velvety white downy mildew sporulation visible on leaf undersides in high humidity',
      'Dark brown or black necrotic girdling on petioles and stems',
      'Rapid canopy collapse with foul decaying organic odor',
      'Granular reddish-brown dry rot extending into tuber flesh beneath the skin',
    ],
    riskFactors: [
      'Use of non-certified seed potatoes or unmonitored backyard seed sharing',
      'Nearby unmanaged potato cull piles or volunteer potato sprouts in compost',
      'Cool rainy weather combined with overhead sprinkler irrigation',
    ],
    immediateActions: [
      'Bag and destroy heavily blighted plants immediately — never compost late-blight foliage.',
      'Apply systemic protectant fungicides (mancozeb, chlorothalonil, or copper) to unaffected adjacent foliage.',
      'Hill up potato rows with an extra 4 inches of soil to shield tubers from spore washdown.',
      'Terminate diseased vines 2-3 weeks before harvest to prevent tuber contamination at digging.',
    ],
    preventiveActions: [
      'Always plant state-certified disease-free seed tubers.',
      'Destroy all potato cull piles and volunteer sprouts before springtime emergence.',
      'Sign up for regional late blight tracking alerts and scout fields twice weekly.',
      'Ensure hill shoulders are wide and well-packed above growing tubers.',
    ],
    cropCarePlan: [
      { action: 'Isolate & rogue out infected plants in sealed bags', priority: 'High', time: 'Immediately (Day 1)' },
      { action: 'Apply foliar protectant spray to surrounding rows', priority: 'High', time: 'Within 12 Hours' },
      { action: 'Build up row hilling to protect underground tubers', priority: 'High', time: 'Within 2 Days' },
      { action: 'Check local agricultural extension alert networks', priority: 'Medium', time: 'Daily' },
      { action: 'Vine kill 14 days before harvest if blight present', priority: 'High', time: 'Pre-Harvest' },
    ],
  },

  healthy_leaf: {
    diseaseKey: 'healthy_leaf',
    canonicalClass: 'general__healthy',
    plant: 'Crop Foliage',
    disease: 'Healthy Leaf Canopy',
    pathogen: 'None (Healthy Living Tissue)',
    category: 'Healthy Condition',
    severity: 'None (Healthy)',
    severityScore: 4,
    confidence: 0.992,
    probabilities: {
      'Healthy Foliage': 0.992,
      'Early Blight Trace': 0.004,
      'Nutrient Chlorosis Trace': 0.003,
      'Leaf Spot Trace': 0.001,
    },
    description:
      'The examined leaf sample displays optimal physiological vigor: uniform chlorophyll distribution, intact leaf margins, crisp venation, and zero macroscopic fungal pustules, bacterial spots, or necrotic lesioning.',
    cause:
      'Balanced plant nutrition, clean seed stock, optimal photosynthesis, and disciplined crop care.',
    transmission: 'Not Applicable — Leaf is healthy.',
    favorableConditions:
      'Adequate sunlight, steady root-zone moisture, good air circulation, and balanced macro/micronutrient soil availability.',
    symptoms: [
      'Uniform vibrant green coloration across entire lamina and petiole',
      'Intact, smooth leaf margins without chlorotic yellowing or necrotic crisping',
      'Strong turgidity with no wilting or drooping symptoms',
      'Zero visible fungal spore fruiting bodies or powdery mildew coats',
      'Healthy vascular veining supporting active transpiration and photosynthesis',
    ],
    riskFactors: [
      'Prolonged drought or waterlogged soil conditions could weaken root defense',
      'Sudden pest pressure (aphids, whiteflies) could introduce viral vectors',
    ],
    immediateActions: [
      'Maintain existing balanced watering and fertilization schedules.',
      'Record current healthy baseline imagery for comparative seasonal tracking.',
      'Keep mulch fresh and inspect lower foliage periodically.',
    ],
    preventiveActions: [
      'Perform regular bi-weekly scouting for early signs of foliar stress.',
      'Ensure adequate spacing as the canopy continues to expand.',
      'Sanitize all garden shears and weeding implements routinely.',
    ],
    cropCarePlan: [
      { action: 'Continue balanced irrigation & organic feeding', priority: 'Low', time: 'Ongoing' },
      { action: 'Perform routine bi-weekly foliar scouting', priority: 'Low', time: 'Every 2 Weeks' },
      { action: 'Maintain tool sanitation and weeding', priority: 'Low', time: 'Monthly' },
    ],
  },

  tomato_late_blight: {
    diseaseKey: 'tomato_late_blight',
    canonicalClass: 'tomato__late_blight',
    plant: 'Tomato',
    disease: 'Late Blight',
    pathogen: 'Phytophthora infestans',
    category: 'Oomycete',
    severity: 'High',
    severityScore: 92,
    confidence: 0.975,
    probabilities: {
      'Tomato — Late Blight': 0.975,
      'Tomato — Early Blight': 0.016,
      'Tomato — Leaf Mold': 0.006,
      'Tomato — Healthy': 0.003,
    },
    description:
      'Tomato late blight is a rapid, devastating water mold disease causing large irregular water-soaked dark blotches across leaves and stems, with white downy sporangia on leaf undersides in damp weather.',
    cause: 'Pathogen Phytophthora infestans thriving in cool, damp conditions.',
    transmission: 'Airborne sporangia carried long distances by wind and rain.',
    favorableConditions: 'Temperatures 15°C - 22°C with high relative humidity (>90%).',
    symptoms: [
      'Rapidly enlarging water-soaked pale green to dark brown lesions',
      'Delicate white fungal-like fuzz on undersides of affected leaves during high humidity',
      'Stem girdling leading to sudden branch collapse',
      'Firm, dark greasy rot on tomato fruits',
    ],
    riskFactors: ['Cool foggy weather', 'Overhead watering', 'Infected potatoes nearby'],
    immediateActions: [
      'Immediately remove and bag heavily infected plants.',
      'Apply protective copper or systemic fungicide across uninfected rows.',
      'Stop overhead watering immediately.',
    ],
    preventiveActions: [
      'Plant resistant tomato varieties such as Defiant PhR or Mountain Merit.',
      'Space plants generously to promote rapid canopy drying.',
    ],
    cropCarePlan: [
      { action: 'Rogue out severely diseased plants', priority: 'High', time: 'Immediately' },
      { action: 'Foliar fungicide application', priority: 'High', time: 'Within 24 Hours' },
      { action: 'Daily field inspection', priority: 'Medium', time: 'Daily' },
    ],
  },

  apple_scab: {
    diseaseKey: 'apple_scab',
    canonicalClass: 'apple__apple_scab',
    plant: 'Apple',
    disease: 'Apple Scab',
    pathogen: 'Venturia inaequalis',
    category: 'Fungal Ascomycota',
    severity: 'Moderate',
    severityScore: 65,
    confidence: 0.965,
    probabilities: {
      'Apple — Apple Scab': 0.965,
      'Apple — Black Rot': 0.021,
      'Apple — Cedar Rust': 0.010,
      'Apple — Healthy': 0.004,
    },
    description:
      'Apple scab causes olive-green to dark brown velvety spots on apple leaves and fruit, causing puckering, leaf drop, and corky scab lesions on developing apples.',
    cause: 'Fungus Venturia inaequalis overwintering in fallen leaves.',
    transmission: 'Ascospores ejected from leaf litter in spring rain events.',
    favorableConditions: 'Cool, wet spring weather (13°C - 24°C) with persistent leaf wetness.',
    symptoms: [
      'Olive-green to dull dark brown velvety spots with fuzzy margins',
      'Leaves become puckered and distorted, dropping prematurely',
      'Scabby, corky brown lesions on fruit skin',
    ],
    riskFactors: ['Unraked leaf litter', 'Dense unpruned canopy', 'Prolonged spring rains'],
    immediateActions: [
      'Prune canopy to maximize sunlight penetration and air movement.',
      'Apply protective fungicide spray during early vegetative flush.',
    ],
    preventiveActions: [
      'Rake and shred fallen leaves in autumn to eliminate overwintering spores.',
      'Plant scab-resistant cultivars such as Liberty, Prima, or Freedom.',
    ],
    cropCarePlan: [
      { action: 'Prune water sprouts and interior canopy', priority: 'Medium', time: 'During dormancy' },
      { action: 'Apply protective bio-fungicide during wet spells', priority: 'High', time: 'Spring flush' },
      { action: 'Rake and destroy fallen leaves', priority: 'Medium', time: 'Autumn' },
    ],
  },
};

/**
 * Helper to match an identifier (class_name, disease key, or free text) to a detailed disease record
 */
export function lookupDisease(queryKey) {
  if (!queryKey) return DISEASE_DATABASE.healthy_leaf;
  const normalized = String(queryKey).toLowerCase().replace(/[\s\-—]+/g, '_');

  if (DISEASE_DATABASE[normalized]) return DISEASE_DATABASE[normalized];

  for (const [key, item] of Object.entries(DISEASE_DATABASE)) {
    if (
      normalized.includes(key) ||
      (item.canonicalClass && normalized.includes(item.canonicalClass)) ||
      (item.disease && normalized.includes(item.disease.toLowerCase()))
    ) {
      return item;
    }
  }

  // Fallback heuristic based on keywords
  if (normalized.includes('healthy')) return DISEASE_DATABASE.healthy_leaf;
  if (normalized.includes('rust')) return DISEASE_DATABASE.corn_common_rust;
  if (normalized.includes('late') && normalized.includes('blight')) return DISEASE_DATABASE.potato_late_blight;
  if (normalized.includes('blight')) return DISEASE_DATABASE.tomato_early_blight;
  if (normalized.includes('scab')) return DISEASE_DATABASE.apple_scab;

  return DISEASE_DATABASE.tomato_early_blight;
}
