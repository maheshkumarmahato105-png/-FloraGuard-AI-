"""Agronomic disease knowledge base covering all 38 PlantVillage classes.

Maps raw PlantVillage class identifiers (e.g., 'Tomato___Early_blight') to
structured human-readable plant names, disease names, symptoms, causes,
prevention methods, and treatment recommendations.
"""

from typing import Dict, List, Tuple


def _entry(
    plant: str,
    disease: str,
    is_healthy: bool,
    severity: str,
    description: str,
    symptoms: List[str],
    causes: List[str],
    prevention: List[str],
    treatment: List[str],
) -> Dict:
    return {
        "plant_name": plant,
        "disease_name": disease,
        "is_healthy": is_healthy,
        "severity": severity,
        "description": description,
        "symptoms": symptoms,
        "causes": causes,
        "prevention": prevention,
        "treatment": treatment,
    }


DISEASE_KNOWLEDGE: Dict[str, Dict] = {
    # ===================== APPLE =====================
    "Apple___Apple_scab": _entry(
        plant="Apple",
        disease="Apple Scab",
        is_healthy=False,
        severity="Moderate to High",
        description="A common fungal disease caused by Venturia inaequalis that affects apple leaves and fruit, reducing yield and fruit marketability.",
        symptoms=[
            "Olive-green to dark brown velvety spots on leaf surfaces.",
            "Distorted, curled, or dwarfed leaves with premature leaf drop.",
            "Dark, corky, scabby lesions on developing fruit.",
        ],
        causes=[
            "Fungal pathogen Venturia inaequalis overwintering in fallen infected leaves.",
            "Cool, wet spring weather (13–24°C) that releases and disperses ascospores.",
            "Prolonged leaf wetness and poor canopy airflow.",
        ],
        prevention=[
            "Rake and destroy fallen apple leaves in autumn to remove overwintering inoculum.",
            "Prune trees regularly to open the canopy and accelerate leaf drying.",
            "Plant scab-resistant apple cultivars where feasible.",
            "Avoid overhead irrigation during cool spring periods.",
        ],
        treatment=[
            "Apply protective fungicides (such as captan, sulfur, or myclobutanil) starting at green-tip stage per local extension schedules.",
            "Remove heavily infected shoots and leaves to curb secondary spore spread.",
            "Consult local agricultural extension guidance for resistance-managed spray rotations.",
        ],
    ),
    "Apple___Black_rot": _entry(
        plant="Apple",
        disease="Black Rot",
        is_healthy=False,
        severity="High",
        description="A fungal disease caused by Botryosphaeria obtusa that creates 'frog-eye' leaf spots, fruit rot, and bark cankers on apple trees.",
        symptoms=[
            "Circular purple flecks expanding into tan-centered lesions with purple borders ('frog-eye' leaf spot).",
            "Yellowing and premature drop of heavily spotted leaves.",
            "Firm, dark brown-to-black decaying rings on fruit and sunken limb cankers.",
        ],
        causes=[
            "Fungus Botryosphaeria obtusa surviving in mummified fruit, dead wood, and bark cankers.",
            "Warm, humid weather and rain-splashed spores.",
            "Tree stress or wounds from hail, pruning, or insect damage.",
        ],
        prevention=[
            "Prune out dead branches, fire-blight strikes, and cankered limbs during dormancy.",
            "Remove and destroy mummified fruit left on the tree or orchard floor.",
            "Maintain balanced fertilization and avoid mechanical bark injury.",
        ],
        treatment=[
            "Apply labeled fungicides (e.g., captan or strobilurin-based products) from petal fall through summer.",
            "Sanitize pruning tools between cuts with 70% alcohol or bleach solution.",
            "Consult local agricultural guidance for integrated orchard disease management.",
        ],
    ),
    "Apple___Cedar_apple_rust": _entry(
        plant="Apple",
        disease="Cedar Apple Rust",
        is_healthy=False,
        severity="Moderate",
        description="A heteroecious fungal disease caused by Gymnosporangium juniperi-virginianae requiring both juniper/cedar and apple hosts to complete its life cycle.",
        symptoms=[
            "Bright yellow-orange circular spots on upper leaf surfaces.",
            "Tiny black spore-producing specks (pycnia) inside the orange lesions.",
            "Tubular fringed spore structures (aecia) forming on the underside of leaves.",
        ],
        causes=[
            "Fungus Gymnosporangium juniperi-virginianae spreading via windborne spores from gelatinous galls on Eastern red cedar or juniper trees.",
            "Warm, rainy spring weather during bloom and early leaf expansion.",
        ],
        prevention=[
            "Select rust-resistant apple varieties when establishing new plantings.",
            "Remove nearby Eastern red cedar or juniper galls where practical.",
            "Maintain good airflow through seasonal canopy pruning.",
        ],
        treatment=[
            "Apply preventive fungicides (such as myclobutanil, sulfur, or mancozeb) from pink bud through early summer.",
            "Monitor nearby junipers for orange jelly-like spore horns in spring to time sprays accurately.",
            "Follow local agricultural extension recommendations for labeled fungicide use.",
        ],
    ),
    "Apple___healthy": _entry(
        plant="Apple",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The apple leaf appears healthy with uniform green coloration and no visible signs of fungal, bacterial, or viral lesions.",
        symptoms=[
            "Consistent vibrant green leaf coloration.",
            "Intact leaf margins without necrotic spots, powdery coating, or distortion.",
        ],
        causes=[
            "Balanced nutrition, adequate sunlight, and effective pest/disease management.",
        ],
        prevention=[
            "Continue routine orchard sanitation by clearing fallen leaves each season.",
            "Maintain open canopy pruning and drip/root-zone irrigation.",
            "Inspect leaves periodically during wet spring weather for early signs of scab or rust.",
        ],
        treatment=[
            "No treatment required. Maintain standard preventive cultural care.",
        ],
    ),
    # ===================== BLUEBERRY =====================
    "Blueberry___healthy": _entry(
        plant="Blueberry",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The blueberry foliage appears healthy with normal pigmentation and no visible blight, leaf spot, or scorching symptoms.",
        symptoms=[
            "Clean, uniform green foliage with firm leaf structure.",
            "No necrotic margins, spots, or powdery residues.",
        ],
        causes=[
            "Proper acidic soil pH (4.5–5.5), well-drained root zone, and good cultural practices.",
        ],
        prevention=[
            "Maintain soil pH between 4.5 and 5.5 and mulch with pine bark.",
            "Water at the base of plants and prune old canes annually for airflow.",
        ],
        treatment=[
            "No treatment needed. Continue regular monitoring.",
        ],
    ),
    # ===================== CHERRY =====================
    "Cherry_(including_sour)___Powdery_mildew": _entry(
        plant="Cherry",
        disease="Powdery Mildew",
        is_healthy=False,
        severity="Moderate",
        description="A fungal disease caused by Podosphaera clandestina that covers young cherry leaves and shoots in a white powdery mycelium.",
        symptoms=[
            "White to grayish powdery fungal growth on leaf undersides and upper surfaces.",
            "Upward curling, puckering, and distortion of young expanding leaves.",
            "Stunted shoot growth and premature defoliation in severe cases.",
        ],
        causes=[
            "Fungus Podosphaera clandestina overwintering in dormant buds or fallen leaves.",
            "Warm days, cool humid nights, and dense shaded tree canopies.",
        ],
        prevention=[
            "Prune trees to maximize sunlight penetration and airflow.",
            "Avoid excessive nitrogen fertilization that promotes overly succulent shoot growth.",
            "Remove root suckers and water sprouts where mildew often initiates.",
        ],
        treatment=[
            "Apply horticultural oil, potassium bicarbonate, or labeled sulfur/demethylation-inhibitor fungicides early at symptom onset.",
            "Avoid applying sulfur during high temperatures (>32°C) to prevent foliage burn.",
            "Consult local extension services for cherry-safe spray intervals.",
        ],
    ),
    "Cherry_(including_sour)___healthy": _entry(
        plant="Cherry",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The cherry leaf appears vigorous and free from powdery mildew, leaf spot, or visible pest injury.",
        symptoms=[
            "Smooth, glossy green leaf blade with serrated intact edges.",
            "Absence of white powdery patches or shot-hole lesions.",
        ],
        causes=[
            "Healthy growing conditions, balanced irrigation, and good canopy ventilation.",
        ],
        prevention=[
            "Continue annual dormant pruning and autumn leaf cleanup.",
            "Monitor tender shoots in late spring for early powdery mildew.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== CORN (MAIZE) =====================
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": _entry(
        plant="Corn (Maize)",
        disease="Cercospora Leaf Spot (Gray Leaf Spot)",
        is_healthy=False,
        severity="High",
        description="A major yield-limiting fungal disease of maize caused by Cercospora zeae-maydis producing characteristic rectangular gray-tan lesions.",
        symptoms=[
            "Small pinpoint tan spots that elongate into narrow, rectangular gray-brown lesions bounded by leaf veins.",
            "Lesions coalesce under humid conditions, blighting large areas of lower and middle leaves.",
            "Premature leaf death leading to poor ear fill and stalk lodging.",
        ],
        causes=[
            "Fungus Cercospora zeae-maydis surviving on surface corn residue from prior seasons.",
            "Extended periods of high humidity (>90%), heavy dew, and warm temperatures (22–30°C).",
            "Continuous corn cropping without rotation.",
        ],
        prevention=[
            "Rotate corn with non-host crops such as soybeans or small grains for 1–2 years.",
            "Plant corn hybrids with strong genetic resistance to gray leaf spot.",
            "Manage crop residue through conservation tillage where appropriate.",
        ],
        treatment=[
            "Scout fields prior to tasseling (VT/R1); if lesions progress to the ear leaf on susceptible hybrids, consider a labeled foliar fungicide (strobilurin/triazole mix).",
            "Follow integrated pest management (IPM) thresholds from local extension agronomists.",
        ],
    ),
    "Corn_(maize)___Common_rust_": _entry(
        plant="Corn (Maize)",
        disease="Common Rust",
        is_healthy=False,
        severity="Moderate",
        description="A fungal disease caused by Puccinia sorghi characterized by brick-red pustules scattered across both sides of corn leaves.",
        symptoms=[
            "Small, elongated cinnamon-brown to brick-red powdery pustules (uredinia) on both upper and lower leaf surfaces.",
            "Pustules darken to black late in the season as teliospores form.",
            "Chlorosis or leaf tissue death around dense pustule clusters.",
        ],
        causes=[
            "Windborne urediniospores of Puccinia sorghi arriving from southern overwintering regions.",
            "Cool to moderate temperatures (16–23°C) with high humidity and frequent dew.",
        ],
        prevention=[
            "Select corn hybrids bred with partial or major-gene resistance to common rust.",
            "Plant early within recommended planting windows to avoid peak late-season spore loads.",
        ],
        treatment=[
            "Field corn rarely requires fungicide unless severe infection occurs before tasseling on susceptible hybrids.",
            "For high-value sweet corn or seed corn, apply labeled foliar fungicides at early pustule detection.",
            "Consult local agricultural advisors for economic treatment thresholds.",
        ],
    ),
    "Corn_(maize)___Northern_Leaf_Blight": _entry(
        plant="Corn (Maize)",
        disease="Northern Leaf Blight",
        is_healthy=False,
        severity="High",
        description="A foliar fungal disease caused by Exserohilum turcicum that forms large cigar-shaped gray-green lesions on maize leaves.",
        symptoms=[
            "Long, elliptical ('cigar-shaped') gray-green to tan lesions measuring 2.5 to 15 cm in length.",
            "Dark olive-green fungal sporulation visible in lesion centers during humid weather.",
            "Lesions merge to kill entire leaves, severely reducing photosynthetic capacity.",
        ],
        causes=[
            "Fungus Exserohilum turcicum overwintering in infected corn debris.",
            "Moderate temperatures (18–27°C) combined with heavy dew, fog, or frequent rain.",
        ],
        prevention=[
            "Use hybrids rated highly for Northern Leaf Blight resistance.",
            "Practice crop rotation with non-host crops and incorporate infected residue after harvest.",
            "Maintain balanced soil fertility to support vigorous plant defense.",
        ],
        treatment=[
            "Apply labeled foliar fungicides between VT (tasseling) and R2 if disease reaches the third leaf below the ear on susceptible hybrids.",
            "Consult local extension specialists for fungicide mode-of-action rotation.",
        ],
    ),
    "Corn_(maize)___healthy": _entry(
        plant="Corn (Maize)",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The corn leaf is healthy with rich green tissue, intact parallel venation, and no visible blight lesions or rust pustules.",
        symptoms=[
            "Uniform deep green leaf blade with clean parallel veins.",
            "No cigar-shaped lesions, rectangular gray spots, or rusty pustules.",
        ],
        causes=[
            "Strong hybrid genetics, proper soil nutrition, and low pathogen pressure.",
        ],
        prevention=[
            "Continue scouting lower leaves during warm, humid periods prior to tasseling.",
            "Maintain crop rotation and balanced nitrogen/potassium fertility.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== GRAPE =====================
    "Grape___Black_rot": _entry(
        plant="Grape",
        disease="Black Rot",
        is_healthy=False,
        severity="High",
        description="A destructive fungal disease caused by Guignardia bidwellii affecting grape leaves, shoots, and berries.",
        symptoms=[
            "Small, circular tan-to-reddish-brown leaf spots with sharply defined dark brown borders.",
            "Tiny black pinpoint fruiting bodies (pycnidia) arranged in a ring inside older lesions.",
            "Infected berries shrivel into hard, black raisin-like 'mummies'.",
        ],
        causes=[
            "Fungus Guignardia bidwellii overwintering in mummified berries and infected canes.",
            "Warm, rainy weather (20–30°C) during spring shoot growth and fruit set.",
        ],
        prevention=[
            "Remove and destroy all mummified berries and diseased tendrils during dormant pruning.",
            "Train vines and pull leaves around the fruit zone to maximize airflow and sunlight.",
            "Control weeds beneath the trellis to lower canopy humidity.",
        ],
        treatment=[
            "Apply protective fungicides (e.g., myclobutanil, mancozeb, or captan) from early shoot growth through 4–6 weeks post-bloom.",
            "Follow vineyard IPM guidelines from your local agricultural extension office.",
        ],
    ),
    "Grape___Esca_(Black_Measles)": _entry(
        plant="Grape",
        disease="Esca (Black Measles)",
        is_healthy=False,
        severity="High",
        description="A complex grapevine trunk disease caused by wood-inhabiting fungi (including Phaeomoniella chlamydospora and Phaeoacremonium spp.) causing 'tiger-stripe' leaf scorching.",
        symptoms=[
            "Interveinal yellowing (white cultivars) or reddish-brown scorching (red cultivars) creating a 'tiger-stripe' pattern on leaves.",
            "Necrotic drying along leaf margins and between main veins.",
            "Dark purplish-black spots on berry skins ('black measles') and internal wood decay.",
        ],
        causes=[
            "Vascular trunk-infecting fungi entering primarily through large pruning wounds.",
            "Moisture stress, heat waves, and aging vine stress triggering acute foliar symptoms.",
        ],
        prevention=[
            "Prune vines late in dormancy during dry weather to minimize spore infection of wounds.",
            "Apply wound protectants (sealants or biocontrols like Trichoderma) immediately after large cuts.",
            "Remove and destroy severely symptomatic dead cordons or trunks.",
        ],
        treatment=[
            "There is no curative foliar fungicide for internal trunk wood infection.",
            "Practice remedial trunk surgery (retraining a healthy basal sucker) where viable.",
            "Minimize water and crop-load stress and consult viticulture extension specialists.",
        ],
    ),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": _entry(
        plant="Grape",
        disease="Leaf Blight (Isariopsis Leaf Spot)",
        is_healthy=False,
        severity="Moderate",
        description="A late-season fungal leaf disease caused by Pseudocercospora vitis resulting in irregular dark brown necrotic blotches and premature defoliation.",
        symptoms=[
            "Irregular dark brown to reddish-brown angular lesions scattered across mature leaves.",
            "Yellow halos around expanding necrotic patches that merge into large blighted areas.",
            "Premature leaf drop reducing vine vigor and winter hardiness.",
        ],
        causes=[
            "Fungus Pseudocercospora vitis overwintering in fallen vineyard leaf litter.",
            "Warm, humid late-summer weather and dense unpruned foliage.",
        ],
        prevention=[
            "Collect and compost or destroy fallen grape leaves after harvest.",
            "Manage canopy density via shoot positioning and summer leaf thinning.",
            "Avoid overhead irrigation; use drip irrigation instead.",
        ],
        treatment=[
            "Apply copper-based or broad-spectrum protective fungicides if early outbreaks threaten canopy retention.",
            "Consult local viticulture extension guidance for pre-harvest interval compliance.",
        ],
    ),
    "Grape___healthy": _entry(
        plant="Grape",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The grape leaf shows healthy green color and crisp lobing with no black rot lesions, tiger-stripe scorching, or leaf blight.",
        symptoms=[
            "Vibrant green, turgid leaf tissue with uniform venation.",
            "No necrotic spots, rings, or interveinal scorching.",
        ],
        causes=[
            "Well-managed vineyard canopy, proper pruning hygiene, and balanced irrigation.",
        ],
        prevention=[
            "Maintain good trellis airflow and remove fallen debris after leaf drop.",
            "Inspect foliage weekly during warm wet spells.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== ORANGE =====================
    "Orange___Haunglongbing_(Citrus_greening)": _entry(
        plant="Orange (Citrus)",
        disease="Huanglongbing (Citrus Greening)",
        is_healthy=False,
        severity="Critical",
        description="A devastating bacterial disease caused by Candidatus Liberibacter spp. vectored by the Asian citrus psyllid, causing asymmetric blotchy leaf mottling and bitter, misshapen fruit.",
        symptoms=[
            "Asymmetric blotchy yellowing (mottling) across leaf veins—unlike symmetric nutrient deficiencies.",
            "Yellowing of individual shoots ('yellow dragon' symptom) and thickened, leathery leaves.",
            "Small, lopsided, bitter fruit that remain green at the stylar end.",
        ],
        causes=[
            "Phloem-limited bacterium Candidatus Liberibacter asiaticus.",
            "Transmitted by the Asian citrus psyllid (Diaphorina citri) or infected grafting material.",
        ],
        prevention=[
            "Plant only certified disease-free nursery stock.",
            "Monitor and control Asian citrus psyllid populations strictly.",
            "Never move citrus cuttings or plant material from quarantined zones.",
        ],
        treatment=[
            "There is currently no cure for Huanglongbing; confirmed infected trees should be reported to local agricultural authorities and removed to protect neighboring trees.",
            "Implement area-wide psyllid management and enhanced foliar micronutrient programs under official extension guidance.",
        ],
    ),
    # ===================== PEACH =====================
    "Peach___Bacterial_spot": _entry(
        plant="Peach",
        disease="Bacterial Spot",
        is_healthy=False,
        severity="Moderate to High",
        description="A bacterial disease caused by Xanthomonas arboricola pv. pruni causing angular water-soaked leaf spots that drop out to create a 'shot-hole' appearance.",
        symptoms=[
            "Small, angular water-soaked spots turning purple-brown to black on leaf blades.",
            "Infected tissue drops out leaving ragged holes ('shot-hole' effect).",
            "Severe yellowing and premature defoliation, plus pitted lesions on fruit.",
        ],
        causes=[
            "Bacterium Xanthomonas arboricola pv. pruni overwintering in twig cankers and buds.",
            "Wind-driven rain and sandy soils that cause micro-abrasions on foliage.",
        ],
        prevention=[
            "Plant bacterial-spot-resistant peach cultivars.",
            "Establish windbreaks and avoid planting in exposed, sandy, wind-scoured sites.",
            "Avoid excessive nitrogen that stimulates tender, vulnerable shoots.",
        ],
        treatment=[
            "Apply dormant copper sprays before bud swell to reduce overwintering bacteria.",
            "During the growing season, consult local extension schedules for labeled copper or oxytetracycline programs.",
        ],
    ),
    "Peach___healthy": _entry(
        plant="Peach",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The peach leaf is healthy with clean lance-shaped green tissue and no shot-hole lesions or curl distortion.",
        symptoms=[
            "Uniform green coloration across the elongated leaf blade.",
            "No angular necrotic spots, shot-holes, or curling.",
        ],
        causes=[
            "Good orchard hygiene, resistant genetics, and balanced water/nutrient management.",
        ],
        prevention=[
            "Continue annual dormant pruning and balanced fertilization.",
            "Monitor leaves after wind-driven spring rains.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== PEPPER (BELL) =====================
    "Pepper,_bell___Bacterial_spot": _entry(
        plant="Pepper (Bell)",
        disease="Bacterial Spot",
        is_healthy=False,
        severity="High",
        description="A common bacterial disease caused by Xanthomonas spp. leading to dark water-soaked leaf lesions, defoliation, and sunscald on exposed peppers.",
        symptoms=[
            "Small, irregular water-soaked spots that turn dark brown to black on leaves.",
            "Yellow halos surrounding spots, causing leaves to turn yellow and drop prematurely.",
            "Raised, scabby, corky lesions on pepper fruit.",
        ],
        causes=[
            "Pathogenic Xanthomonas bacteria introduced via contaminated seed, transplants, or crop debris.",
            "Warm (24–30°C), wet conditions and splash dispersal from rain or overhead watering.",
        ],
        prevention=[
            "Use certified disease-free seed and resistant bell pepper varieties.",
            "Rotate peppers and tomatoes away from the same soil for at least 2–3 years.",
            "Use drip irrigation and mulch to eliminate soil splash onto foliage.",
            "Avoid handling or harvesting plants while foliage is wet.",
        ],
        treatment=[
            "Remove and destroy severely infected leaves or plants early to limit spread.",
            "Apply copper-based bactericides (often tank-mixed with mancozeb per label) preventively during warm, rainy weather.",
            "Consult local agricultural extension recommendations for resistance management.",
        ],
    ),
    "Pepper,_bell___healthy": _entry(
        plant="Pepper (Bell)",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The bell pepper leaf appears healthy with glossy green coloration and no bacterial spots, mosaic mottling, or wilting.",
        symptoms=[
            "Smooth, deep green, glossy leaf surface.",
            "No dark water-soaked lesions, yellowing halos, or leaf curling.",
        ],
        causes=[
            "Healthy soil drainage, proper spacing, and splash-free irrigation.",
        ],
        prevention=[
            "Water at the root zone using drip irrigation and maintain organic mulch.",
            "Rotate solanaceous crops annually and inspect undersides of leaves regularly.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== POTATO =====================
    "Potato___Early_blight": _entry(
        plant="Potato",
        disease="Early Blight",
        is_healthy=False,
        severity="Moderate to High",
        description="A widespread fungal disease caused by Alternaria solani that produces dark concentric 'target-board' spots primarily on older potato leaves.",
        symptoms=[
            "Dark brown to black circular lesions containing distinct concentric rings ('bull's-eye' or target pattern).",
            "Yellowing (chlorosis) of leaf tissue surrounding the lesions.",
            "Starts on lower, older foliage and progresses upward, causing premature leaf senescence.",
        ],
        causes=[
            "Fungus Alternaria solani overwintering in soil, plant debris, and infected tubers.",
            "Warm temperatures (24–29°C) with alternating wet and dry periods (heavy dew/rain).",
            "Plant stress from low nitrogen, drought, or heavy tuber set.",
        ],
        prevention=[
            "Practice a 3-year crop rotation avoiding potatoes, tomatoes, and eggplants.",
            "Plant certified disease-free seed tubers and tolerant varieties.",
            "Maintain steady soil moisture and adequate nitrogen/potassium fertility.",
            "Avoid overhead watering late in the day.",
        ],
        treatment=[
            "Remove and dispose of severely blighted lower leaves when practical on small plots.",
            "Apply protectant fungicides (e.g., chlorothalonil, mancozeb, or copper-based products) at first sign of disease per label directions.",
            "Consult local agricultural guidance for resistance-rotated fungicide programs.",
        ],
    ),
    "Potato___Late_blight": _entry(
        plant="Potato",
        disease="Late Blight",
        is_healthy=False,
        severity="Critical",
        description="A fast-spreading, destructive oomycete disease caused by Phytophthora infestans capable of destroying entire potato crops within days under cool, wet conditions.",
        symptoms=[
            "Large, irregular water-soaked pale green to dark brown/black blotches on leaf tips and margins.",
            "Distinctive white, cottony mold growth on the underside of lesions during humid mornings.",
            "Rapid blackened collapse of foliage and stems accompanied by a characteristic musty odor.",
        ],
        causes=[
            "Oomycete pathogen Phytophthora infestans surviving in cull piles, volunteer potatoes, or infected seed tubers.",
            "Cool, damp weather (10–21°C) with persistent rain, fog, or >90% relative humidity.",
        ],
        prevention=[
            "Plant only certified late-blight-free seed tubers and eliminate cull piles and volunteer plants.",
            "Hill soil well over tubers to shield them from rain-washed spores.",
            "Space rows for maximum airflow and avoid evening overhead irrigation.",
            "Monitor regional late blight forecasting alerts.",
        ],
        treatment=[
            "Immediately remove and bag/destroy infected plants in home gardens to protect nearby fields (do not compost).",
            "Apply targeted oomycete fungicides (e.g., chlorothalonil, copper, or specific systemic fungicides) preventively when blight risk is high.",
            "Kill vines 2–3 weeks before tuber harvest to prevent tuber infection; follow local extension guidance urgently.",
        ],
    ),
    "Potato___healthy": _entry(
        plant="Potato",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The potato leaf is vigorous and healthy with uniform green foliage and no signs of early blight rings or late blight water-soaked blotches.",
        symptoms=[
            "Rich green, full-textured leaflets with no necrotic spots or pale halos.",
            "No white fuzzy growth on leaf undersides.",
        ],
        causes=[
            "Clean seed tubers, balanced fertility, and effective moisture management.",
        ],
        prevention=[
            "Continue crop rotation, hill soil around stems, and scout lower leaves after cool rains.",
            "Use drip or furrow irrigation to keep foliage dry.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== RASPBERRY =====================
    "Raspberry___healthy": _entry(
        plant="Raspberry",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The raspberry leaf looks healthy with crisp serrated edges, rich green tissue, and no visible rust, blight, or mosaic symptoms.",
        symptoms=[
            "Clean, bright green compound leaflets with intact venation.",
            "No yellowing, rust pustules, or leaf scorching.",
        ],
        causes=[
            "Good cane thinning, well-drained soil, and proper watering.",
        ],
        prevention=[
            "Remove spent floricanes after fruiting to maintain airflow.",
            "Keep rows weed-free and water at the soil surface.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== SOYBEAN =====================
    "Soybean___healthy": _entry(
        plant="Soybean",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The soybean leaf appears healthy with uniform green coloration and no visible frogeye leaf spot, bacterial blight, or rust.",
        symptoms=[
            "Uniformly green trifoliate leaf tissue without necrotic flecks or chlorosis.",
        ],
        causes=[
            "Favorable field conditions, good soil nodulation, and low pathogen pressure.",
        ],
        prevention=[
            "Continue multi-crop rotation and regular mid-season scouting.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== SQUASH =====================
    "Squash___Powdery_mildew": _entry(
        plant="Squash",
        disease="Powdery Mildew",
        is_healthy=False,
        severity="Moderate",
        description="A ubiquitous fungal disease of cucurbits (caused primarily by Podosphaera xanthii) producing talcum-like white patches across squash leaves.",
        symptoms=[
            "White, powdery, talcum-like spots appearing on upper and lower surfaces of older leaves.",
            "Patches expand to coat entire leaves, causing yellowing, drying, and premature leaf collapse.",
            "Reduced fruit quality and sunscald due to canopy loss.",
        ],
        causes=[
            "Fungal pathogens Podosphaera xanthii or Erysiphe cichoracearum spreading via windborne conidia.",
            "High canopy humidity, dense shaded foliage, and moderate-to-warm temperatures (20–27°C).",
        ],
        prevention=[
            "Grow powdery-mildew-resistant squash varieties.",
            "Space plants generously in full sun to keep air circulating.",
            "Remove and dispose of old heavily coated leaves and post-harvest vine debris.",
        ],
        treatment=[
            "Apply potassium bicarbonate, neem oil, horticultural oil, or sulfur-based fungicides at first white spot detection.",
            "Thoroughly coat both upper and lower leaf surfaces; avoid oils/sulfur during extreme midday heat.",
            "Consult local extension guides for commercial fungicide rotations.",
        ],
    ),
    # ===================== STRAWBERRY =====================
    "Strawberry___Leaf_scorch": _entry(
        plant="Strawberry",
        disease="Leaf Scorch",
        is_healthy=False,
        severity="Moderate",
        description="A fungal disease caused by Diplocarpon earlianum that creates irregular purplish blotches which coalesce, giving strawberry foliage a dried, scorched appearance.",
        symptoms=[
            "Numerous small, irregular dark purple-to-reddish blotches on upper leaf surfaces (without distinct white centers).",
            "Blotches merge until the leaf turns reddish-brown, dries out, and curls upward ('scorched' look).",
            "Weakened plant vigor and reduced berry production in subsequent seasons.",
        ],
        causes=[
            "Fungus Diplocarpon earlianum overwintering on infected living and dead strawberry leaves.",
            "Rain splash, overhead irrigation, and crowded plantings with prolonged moisture.",
        ],
        prevention=[
            "Plant resistant strawberry cultivars and certified clean transplants.",
            "Renovate matted rows after harvest and remove old, dead foliage.",
            "Use drip irrigation and straw mulch to reduce splash dispersal.",
            "Control weeds to improve air movement around crowns.",
        ],
        treatment=[
            "Remove heavily scorched leaves to limit spore production.",
            "Apply labeled protectant fungicides (e.g., captan or copper-based products) during cool, wet spring and autumn growth periods.",
            "Follow local agricultural extension recommendations for safe pre-harvest intervals.",
        ],
    ),
    "Strawberry___healthy": _entry(
        plant="Strawberry",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The strawberry leaf is healthy with bright green leaflets, clean serrated edges, and no purple scorch blotches or powdery mildew.",
        symptoms=[
            "Uniformly green, firm trifoliate leaflets.",
            "No purple blotching, margin browning, or white fungal coating.",
        ],
        causes=[
            "Well-drained beds, drip watering, and clean mulched soil.",
        ],
        prevention=[
            "Maintain straw mulch, proper plant spacing, and periodic post-harvest bed renovation.",
        ],
        treatment=[
            "No treatment required.",
        ],
    ),
    # ===================== TOMATO =====================
    "Tomato___Bacterial_spot": _entry(
        plant="Tomato",
        disease="Bacterial Spot",
        is_healthy=False,
        severity="High",
        description="A bacterial infection caused by Xanthomonas spp. causing dark water-soaked leaf lesions, leaf yellowing, and scabby fruit spots.",
        symptoms=[
            "Small, dark brown to black water-soaked circular/angular spots on leaves.",
            "Yellowing around clustered lesions leading to leaf blight and premature defoliation.",
            "Raised black scabby spots with pale halos on green tomato fruit.",
        ],
        causes=[
            "Xanthomonas bacteria introduced via infected seed, transplants, or soil debris.",
            "Warm, humid, rainy weather and splash dispersal from overhead watering.",
        ],
        prevention=[
            "Use certified disease-free seeds/transplants and practice a 2–3 year crop rotation.",
            "Mulch soil and water exclusively at the base with drip irrigation.",
            "Avoid pruning or handling wet plants; sanitize tools regularly.",
        ],
        treatment=[
            "Prune and discard heavily infected lower leaves immediately (do not compost).",
            "Apply copper-based bactericides (often combined with mancozeb per local label) preventively during humid weather.",
            "Consult local agricultural extension guidance for current resistant varieties and spray rules.",
        ],
    ),
    "Tomato___Early_blight": _entry(
        plant="Tomato",
        disease="Early Blight",
        is_healthy=False,
        severity="Moderate to High",
        description="A very common fungal disease caused by Alternaria solani characterized by dark brown concentric target-like spots beginning on older lower tomato leaves.",
        symptoms=[
            "Brown to dark black circular spots with concentric 'target-ring' patterns on leaves.",
            "Yellowing (chlorosis) spreading outward around infected areas.",
            "Gradual lower-to-upper leaf deterioration and premature defoliation.",
        ],
        causes=[
            "Fungus Alternaria solani overwintering in infected plant residue and soil.",
            "Warm temperatures (24–29°C), high humidity, heavy dew, and splash from soil.",
            "Weakened plants under nutrient stress or heavy fruit load.",
        ],
        prevention=[
            "Rotate tomatoes out of the same bed for at least 3 years.",
            "Stake/cage plants, prune lower leaves near the soil, and apply organic mulch.",
            "Avoid overhead watering; water early in the morning at the root zone.",
            "Space plants properly to improve air circulation.",
        ],
        treatment=[
            "Remove severely infected lower leaves promptly and dispose of them away from the garden.",
            "Apply labeled protectant fungicides (such as chlorothalonil, copper, or biofungicides like Bacillus subtilis) at early symptoms.",
            "Consult local agricultural guidance for appropriate integrated treatment schedules.",
        ],
    ),
    "Tomato___Late_blight": _entry(
        plant="Tomato",
        disease="Late Blight",
        is_healthy=False,
        severity="Critical",
        description="An aggressive oomycete disease caused by Phytophthora infestans that rapidly blights tomato foliage, stems, and fruit under cool, wet conditions.",
        symptoms=[
            "Large, irregular water-soaked greenish-black blotches on leaves and dark brown stem lesions.",
            "Fuzzy white fungal-like sporulation along lesion edges on leaf undersides in damp conditions.",
            "Rapid collapse of foliage and firm, greasy brown rot on green or ripe fruit.",
        ],
        causes=[
            "Oomycete Phytophthora infestans carried on windborne spores from infected potatoes or tomatoes.",
            "Cool, wet weather (10–21°C) with prolonged leaf wetness and high humidity.",
        ],
        prevention=[
            "Grow late-blight-resistant tomato cultivars where available.",
            "Eliminate nearby volunteer potato/tomato plants and avoid overhead watering.",
            "Ensure wide spacing and staking for rapid foliage drying.",
        ],
        treatment=[
            "Remove and seal infected plants in disposal bags immediately to prevent airborne spread to neighboring crops.",
            "Apply preventive fungicides (chlorothalonil, copper, or labeled oomycete products) when regional blight alerts are active.",
            "Seek local agricultural extension guidance immediately upon confirmation.",
        ],
    ),
    "Tomato___Leaf_Mold": _entry(
        plant="Tomato",
        disease="Leaf Mold",
        is_healthy=False,
        severity="Moderate",
        description="A fungal disease caused by Passalora fulva (formerly Fulvia fulva) prevalent in greenhouses, high tunnels, and humid gardens.",
        symptoms=[
            "Pale green to bright yellow irregular patches on upper leaf surfaces.",
            "Distinctive olive-green to grayish-purple velvety mold growth directly underneath those patches.",
            "Infected leaves curl, wither, and drop prematurely while blossoms may abort.",
        ],
        causes=[
            "Fungus Passalora fulva thriving at relative humidity above 85% and temperatures of 21–24°C.",
            "Poor ventilation, crowded foliage, and condensation on leaves.",
        ],
        prevention=[
            "Ventilate greenhouses and high tunnels aggressively to keep relative humidity below 85%.",
            "Choose leaf-mold-resistant tomato cultivars.",
            "Prune suckers, widen plant spacing, and water only at soil level in the morning.",
        ],
        treatment=[
            "Remove affected lower leaves carefully to minimize spore dispersal.",
            "Apply labeled fungicides (e.g., copper, chlorothalonil, or potassium bicarbonate) focusing on leaf undersides.",
            "Sanitize greenhouse surfaces and stakes between growing cycles.",
        ],
    ),
    "Tomato___Septoria_leaf_spot": _entry(
        plant="Tomato",
        disease="Septoria Leaf Spot",
        is_healthy=False,
        severity="Moderate to High",
        description="A destructive fungal foliage disease caused by Septoria lycopersici producing numerous small circular spots with pale gray centers and dark borders.",
        symptoms=[
            "Numerous tiny (1.5–3 mm) circular spots with dark brown borders and tan/light-gray centers.",
            "Tiny black speck-like fruiting bodies (pycnidia) visible in the center of mature spots.",
            "Heavily spotted lower leaves turn yellow, brown, and drop rapidly upward along the stem.",
        ],
        causes=[
            "Fungus Septoria lycopersici surviving on tomato debris, weeds (nightshade), and stakes.",
            "Warm wet weather (20–25°C) with rain splash or overhead irrigation.",
        ],
        prevention=[
            "Remove all tomato debris at season end and disinfect cages/stakes.",
            "Apply thick mulch under plants and avoid overhead watering.",
            "Rotate away from tomatoes for 3 years and control solanaceous weeds.",
        ],
        treatment=[
            "Prune off infected lower leaves early and discard away from the growing area.",
            "Apply protectant fungicides (chlorothalonil, mancozeb, or copper) on a 7–10 day schedule during wet weather.",
            "Follow local extension guidance for organic or conventional spray options.",
        ],
    ),
    "Tomato___Spider_mites Two-spotted_spider_mite": _entry(
        plant="Tomato",
        disease="Spider Mites (Two-Spotted Spider Mite)",
        is_healthy=False,
        severity="Moderate",
        description="An arthropod pest infestation caused by Tetranychus urticae that feeds on leaf cell sap, producing fine stippling, bronzing, and delicate webbing.",
        symptoms=[
            "Fine yellow or white pinpoint stippling scattered across upper leaf surfaces.",
            "Leaves become bronzed, dry, and brittle; severe infestations show fine silken webbing underneath leaves and at shoot tips.",
            "Tiny moving mites (often with two dark body spots) visible on leaf undersides with a hand lens.",
        ],
        causes=[
            "Two-spotted spider mite (Tetranychus urticae) populations exploding during hot, dry, dusty conditions.",
            "Water-stressed plants or broad-spectrum insecticide use that kills natural mite predators.",
        ],
        prevention=[
            "Keep plants well-watered to avoid drought stress and reduce surrounding dust.",
            "Preserve beneficial predators (lacewings, predatory mites, lady beetles) by avoiding unnecessary broad-spectrum insecticides.",
            "Inspect leaf undersides regularly during hot, dry spells.",
        ],
        treatment=[
            "Spray strong jets of water on leaf undersides to dislodge mites and webbing.",
            "Apply insecticidal soap, horticultural oil, or neem oil thoroughly to leaf undersides every 5–7 days.",
            "Consider releasing predatory mites (Phytoseiulus persimilis) in greenhouse settings or consult local extension IPM guidance.",
        ],
    ),
    "Tomato___Target_Spot": _entry(
        plant="Tomato",
        disease="Target Spot",
        is_healthy=False,
        severity="Moderate to High",
        description="A fungal disease caused by Corynespora cassiicola creating sunken brown spots with concentric rings on tomato leaves, stems, and fruit.",
        symptoms=[
            "Small pinpoint water-soaked specks expanding into light-to-dark brown circular lesions with subtle concentric rings.",
            "Yellow halos around lesions; centers often crack or drop out.",
            "Progressive defoliation starting in the lower canopy and pitted lesions on fruit.",
        ],
        causes=[
            "Fungus Corynespora cassiicola surviving in crop debris and alternative weed hosts.",
            "Warm temperatures (20–32°C) with extended high humidity and prolonged leaf wetness.",
        ],
        prevention=[
            "Improve airflow with proper plant spacing, trellising, and lower-leaf pruning.",
            "Use drip irrigation and plastic/organic mulch to minimize moisture and splash.",
            "Rotate crops and clear all crop residue immediately after harvest.",
        ],
        treatment=[
            "Remove heavily affected lower leaves to reduce inoculum load.",
            "Apply labeled protective fungicides (e.g., chlorothalonil, mancozeb, or strobilurin/SDHI rotations) at early disease detection.",
            "Consult local agricultural extension recommendations for resistance management.",
        ],
    ),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": _entry(
        plant="Tomato",
        disease="Tomato Yellow Leaf Curl Virus (TYLCV)",
        is_healthy=False,
        severity="Critical",
        description="A serious begomovirus transmitted by silverleaf whiteflies (Bemisia tabaci) causing severe upward leaf curling, yellowing margins, and stunted growth.",
        symptoms=[
            "Severe upward cupping and curling of young leaves with bright yellow leaf margins.",
            "Markedly stunted, bushy plant growth with reduced leaf size.",
            "Heavy flower drop and drastically reduced or absent fruit set.",
        ],
        causes=[
            "Tomato Yellow Leaf Curl Virus (TYLCV) transmitted exclusively by feeding whiteflies (Bemisia tabaci).",
            "Warm climates and nearby infected host plants or weeds harboring whiteflies.",
        ],
        prevention=[
            "Plant TYLCV-resistant tomato varieties (look for Ty-gene resistance).",
            "Use fine insect-exclusion mesh in nurseries/greenhouses and reflective silver mulches to repel whiteflies.",
            "Control weeds around field margins and install yellow sticky traps to monitor whiteflies.",
        ],
        treatment=[
            "No chemical cure exists for viral infections; promptly rogue (uproot and bag) infected plants early in the season to stop whitefly transmission.",
            "Manage whitefly vectors using insecticidal soap, neem oil, or extension-recommended targeted treatments.",
            "Sanitize hands and tools and consult local agricultural guidance for area-wide vector management.",
        ],
    ),
    "Tomato___Tomato_mosaic_virus": _entry(
        plant="Tomato",
        disease="Tomato Mosaic Virus (ToMV)",
        is_healthy=False,
        severity="High",
        description="A highly contagious, mechanically transmitted tobamovirus causing light-and-dark green mottling, leaf distortion, and mottled fruit.",
        symptoms=[
            "Distinctive light green and dark green mosaic mottling across leaflets.",
            "Narrowed, twisted, fern-like ('shoestring') or blistered young leaves.",
            "Stunted growth and uneven ripening or internal browning of fruit.",
        ],
        causes=[
            "Tomato Mosaic Virus (ToMV) surviving on contaminated seeds, soil debris, tools, hands, and tobacco products.",
            "Spread rapidly through pruning, staking, handling, and plant-to-plant contact.",
        ],
        prevention=[
            "Select ToMV-resistant tomato cultivars (Tm-2² resistance) and certified virus-free seed.",
            "Wash hands thoroughly with soap and water (and avoid tobacco use) before handling plants.",
            "Disinfect pruning shears and stakes in 10% bleach or trisodium phosphate (TSP) solution.",
        ],
        treatment=[
            "No antiviral treatment cures infected plants; remove and bag symptomatic plants carefully without brushing healthy foliage.",
            "Do not compost infected material and avoid planting tomatoes in the same soil the following season.",
            "Follow strict sanitation protocols recommended by local agricultural extension offices.",
        ],
    ),
    "Tomato___healthy": _entry(
        plant="Tomato",
        disease="Healthy",
        is_healthy=True,
        severity="None",
        description="The tomato leaf appears healthy with rich green coloration, well-formed leaflets, and no visible blight, mold, viral mottling, or pest stippling.",
        symptoms=[
            "Deep, even green leaf coloration across leaflets.",
            "No concentric spots, yellowing halos, webbing, or upward margin curling.",
        ],
        causes=[
            "Consistent root-zone watering, balanced nutrients, and good airflow.",
        ],
        prevention=[
            "Continue mulching, staking, and watering at soil level early in the morning.",
            "Inspect lower leaves weekly for early signs of blight or insects.",
        ],
        treatment=[
            "No treatment required. Keep up preventive cultural practices.",
        ],
    ),
}


import re


def _normalize_key(label: str) -> str:
    """Normalize a class label for flexible dictionary lookups."""
    cleaned = (
        label.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(",", "")
        .replace("(", "")
        .replace(")", "")
    )
    return re.sub(r"_+", "_", cleaned).strip("_")


_NORMALIZED_LOOKUP: Dict[str, str] = {
    _normalize_key(k): k for k in DISEASE_KNOWLEDGE.keys()
}


def parse_class_label(raw_label: str) -> Tuple[str, str, bool]:
    """Parse a raw PlantVillage class label into (plant_name, disease_name, is_healthy)."""
    info = get_disease_info(raw_label)
    return info["plant_name"], info["disease_name"], info["is_healthy"]


def get_disease_info(raw_label: str) -> Dict:
    """Retrieve structured disease knowledge for a raw model class label.

    Falls back gracefully if a custom dataset adds a class not listed above.
    """
    if raw_label in DISEASE_KNOWLEDGE:
        return DISEASE_KNOWLEDGE[raw_label]

    norm = _normalize_key(raw_label)
    if norm in _NORMALIZED_LOOKUP:
        return DISEASE_KNOWLEDGE[_NORMALIZED_LOOKUP[norm]]

    # Graceful generic parser for PlantVillage-style 'Plant___Disease' strings
    if "___" in raw_label:
        raw_plant, raw_disease = raw_label.split("___", 1)
    elif "__" in raw_label:
        raw_plant, raw_disease = raw_label.split("__", 1)
    else:
        raw_plant, raw_disease = raw_label, "Unknown Condition"

    plant_name = raw_plant.replace("_", " ").strip().title()
    disease_clean = raw_disease.replace("_", " ").strip()
    is_healthy = "healthy" in disease_clean.lower()
    disease_name = "Healthy" if is_healthy else disease_clean.title()

    if is_healthy:
        return _entry(
            plant=plant_name,
            disease="Healthy",
            is_healthy=True,
            severity="None",
            description=f"The {plant_name} leaf appears healthy with no detected disease symptoms.",
            symptoms=["Uniform green foliage with no prominent necrotic lesions."],
            causes=["Good growing conditions and low pathogen pressure."],
            prevention=["Maintain balanced watering, airflow, and routine visual inspection."],
            treatment=["No treatment required."],
        )

    return _entry(
        plant=plant_name,
        disease=disease_name,
        is_healthy=False,
        severity="Moderate",
        description=f"Detected {disease_name} on {plant_name} foliage based on visual leaf characteristics.",
        symptoms=[
            "Visible discoloration, spotting, or lesion patterns on the leaf surface.",
            "Potential yellowing or deterioration around affected areas.",
        ],
        causes=[
            "Pathogenic infection (fungal, bacterial, or viral) or environmental stress.",
            "Favorable moisture, humidity, or vector conditions.",
        ],
        prevention=[
            "Avoid overhead watering and improve air circulation around plants.",
            "Remove fallen infected plant debris and rotate crops between seasons.",
        ],
        treatment=[
            "Isolate or prune severely affected leaves to slow spread.",
            "Consult local agricultural extension guidance for targeted treatment.",
        ],
    )


def get_supported_plants_summary(class_names: List[str]) -> List[Dict]:
    """Group model class labels by plant for UI presentation."""
    grouped: Dict[str, List[Dict]] = {}
    for raw in class_names:
        info = get_disease_info(raw)
        plant = info["plant_name"]
        grouped.setdefault(plant, []).append(
            {
                "raw_label": raw,
                "disease_name": info["disease_name"],
                "is_healthy": info["is_healthy"],
                "severity": info["severity"],
            }
        )

    result = []
    for plant in sorted(grouped.keys()):
        conditions = sorted(
            grouped[plant], key=lambda item: (item["is_healthy"], item["disease_name"])
        )
        result.append(
            {
                "plant_name": plant,
                "condition_count": len(conditions),
                "conditions": conditions,
            }
        )
    return result
