"""Structured educational plant disease knowledge base (Phase 7).

Key rules:
  - Kept completely separate from model class probabilities.
  - Clearly distinguishes general educational guidance from professional agricultural diagnosis.
  - Never invents chemical pesticide dosages or prescription formulas.
  - Never returns disease treatment advice for uncertain/rejected predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

EDUCATIONAL_DISCLAIMER = (
    "General educational information only — not a professional agricultural diagnosis. "
    "Consult a certified agronomist or local agricultural extension service before applying treatments."
)


def _info(
    disease_name: str,
    affected_plant: str,
    is_healthy: bool,
    summary: str,
    symptoms: List[str],
    prevention: List[str],
    treatment: List[str],
) -> Dict[str, Any]:
    return {
        "disease_name": disease_name,
        "affected_plant": affected_plant,
        "is_healthy": is_healthy,
        "summary": summary,
        "symptoms": symptoms,
        "prevention": prevention,
        "treatment": treatment,
        "disclaimer": EDUCATIONAL_DISCLAIMER,
    }


def _healthy_info(plant: str) -> Dict[str, Any]:
    return _info(
        disease_name="Healthy",
        affected_plant=plant,
        is_healthy=True,
        summary=f"The {plant} leaf shows uniform green coloration without prominent visible lesions or chlorosis.",
        symptoms=[
            "Consistent healthy green coloration across the leaf blade.",
            "Intact leaf margins with no visible necrotic spots, powdery film, or mosaic distortion.",
        ],
        prevention=[
            "Continue balanced root-zone watering and avoid prolonged overhead leaf wetness.",
            "Maintain good airflow through appropriate plant spacing and seasonal sanitation.",
            "Inspect both upper and lower leaf surfaces periodically for early stress signs.",
        ],
        treatment=[
            "No disease intervention required; continue preventive cultural practices.",
        ],
    )


_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "apple__apple_scab": _info(
        "Apple Scab", "Apple", False,
        "Fungal leaf and fruit disease associated with Venturia inaequalis in cool, wet spring conditions.",
        ["Olive-green to dark brown velvety lesions on leaves.", "Leaf curling, distortion, and early defoliation."],
        ["Rake and remove fallen leaves in autumn.", "Prune canopy to improve airflow and speed drying.", "Prefer scab-resistant cultivars when replanting."],
        ["Prune heavily affected shoots where practical.", "Consult local orchard extension guidance for appropriate protective management programs."],
    ),
    "apple__black_rot": _info(
        "Black Rot", "Apple", False,
        "Fungal infection caused by Botryosphaeria obtusa causing characteristic 'frog-eye' leaf spots.",
        ["Circular tan-centered lesions ringed by purple borders ('frog-eye' spots).", "Yellowing and premature drop of heavily spotted leaves."],
        ["Remove dead wood, limb cankers, and mummified fruit during dormancy.", "Sanitize pruning tools between cuts."],
        ["Eliminate overwintering inoculum sources in the orchard.", "Consult local agricultural extension specialists for integrated orchard control."],
    ),
    "apple__cedar_apple_rust": _info(
        "Cedar Apple Rust", "Apple", False,
        "Heteroecious fungal disease alternating between cedar/juniper hosts and apple foliage.",
        ["Bright yellow-orange spots on upper leaf surfaces.", "Small dark specks inside orange spots and raised structures on leaf undersides."],
        ["Select rust-resistant apple varieties.", "Maintain canopy airflow and monitor nearby junipers for spring galls."],
        ["Follow local extension timing guidance for preventive spring management.", "Remove severely affected non-essential twigs if localized."],
    ),
    "apple__healthy": _healthy_info("Apple"),
    "blueberry__healthy": _healthy_info("Blueberry"),
    "cherry__healthy": _healthy_info("Cherry"),
    "cherry__powdery_mildew": _info(
        "Powdery Mildew", "Cherry", False,
        "Fungal foliage disease (Podosphaera clandestina) producing powdery white growth on young shoots and leaves.",
        ["White powdery patches on leaf undersides and upper surfaces.", "Upward curling and puckering of young leaves."],
        ["Prune to increase sunlight penetration and airflow.", "Avoid excessive nitrogen fertilization that drives overly succulent growth."],
        ["Remove infected water sprouts and suckers.", "Consult local tree-fruit extension recommendations for labeled mildew management."],
    ),
    "corn__cercospora_gray_leaf_spot": _info(
        "Cercospora Leaf Spot (Gray Leaf Spot)", "Corn (Maize)", False,
        "Fungal foliar disease caused by Cercospora zeae-maydis favored by warm, humid conditions.",
        ["Narrow rectangular gray-tan lesions constrained by parallel leaf veins.", "Lesions coalesce on lower leaves and progress upward."],
        ["Rotate corn with non-host crops.", "Select hybrids rated for strong gray leaf spot tolerance.", "Manage surface crop residue."],
        ["Scout prior to tasseling and consult local agronomy extension thresholds before any foliar intervention."],
    ),
    "corn__common_rust": _info(
        "Common Rust", "Corn (Maize)", False,
        "Fungal disease caused by Puccinia sorghi producing powdery rust pustules on both leaf surfaces.",
        ["Small elongated cinnamon-brown to brick-red pustules on upper and lower leaves.", "Yellowing around dense pustule clusters."],
        ["Use rust-tolerant corn hybrids.", "Plant within recommended regional planting windows."],
        ["Monitor progression relative to growth stage and consult local extension specialists for economic thresholds."],
    ),
    "corn__healthy": _healthy_info("Corn (Maize)"),
    "corn__northern_leaf_blight": _info(
        "Northern Leaf Blight", "Corn (Maize)", False,
        "Fungal disease caused by Exserohilum turcicum characterized by large cigar-shaped lesions.",
        ["Long elliptical gray-green to tan ('cigar-shaped') leaf lesions.", "Dark sporulation in lesion centers under humid weather."],
        ["Plant resistant hybrids and practice multi-crop rotation.", "Manage infected crop debris after harvest."],
        ["Scout ear-leaf proximity around tasseling and consult local extension advisors for management options."],
    ),
    "grape__black_rot": _info(
        "Black Rot", "Grape", False,
        "Fungal vineyard disease (Guignardia bidwellii) affecting leaves, shoots, and berries.",
        ["Circular tan-brown leaf lesions with dark margins and pinpoint black fruiting bodies.", "Shriveled dark berry mummies on clusters."],
        ["Remove mummified fruit and diseased canes during dormant pruning.", "Open trellis canopy for airflow and sunlight."],
        ["Practice strict vineyard sanitation and consult regional viticulture extension guides for seasonal management."],
    ),
    "grape__esca_black_measles": _info(
        "Esca (Black Measles)", "Grape", False,
        "Complex fungal trunk disease causing characteristic interveinal 'tiger-stripe' leaf scorching.",
        ["Interveinal yellowing or reddish-brown scorching creating a tiger-stripe pattern.", "Marginal leaf drying under heat or water stress."],
        ["Prune late in dormancy during dry conditions.", "Protect large pruning wounds and remove dead wood."],
        ["Foliar sprays do not cure internal trunk infection; consult viticulture extension specialists regarding remedial trunk renewal."],
    ),
    "grape__healthy": _healthy_info("Grape"),
    "grape__leaf_blight_isariopsis": _info(
        "Leaf Blight (Isariopsis Leaf Spot)", "Grape", False,
        "Late-season fungal leaf disease (Pseudocercospora vitis) producing dark irregular blotches.",
        ["Irregular dark brown necrotic patches often ringed by yellowing.", "Premature defoliation of mature canopy leaves."],
        ["Clear fallen leaf litter after harvest.", "Use drip irrigation and maintain open vine canopies."],
        ["Consult local viticulture extension guidance if early outbreaks threaten canopy health."],
    ),
    "orange__huanglongbing_citrus_greening": _info(
        "Huanglongbing (Citrus Greening)", "Orange (Citrus)", False,
        "Serious phloem-limited bacterial disease vectored by the Asian citrus psyllid.",
        ["Asymmetric blotchy yellow mottling across leaf veins.", "Thickened leathery leaves and misshapen bitter fruit."],
        ["Plant only certified disease-free nursery stock.", "Monitor and manage psyllid vectors strictly; respect quarantine zones."],
        ["No cure currently exists; report suspected cases to local agricultural authorities for official confirmatory testing and guidance."],
    ),
    "peach__bacterial_spot": _info(
        "Bacterial Spot", "Peach", False,
        "Bacterial foliage and fruit disease (Xanthomonas arboricola pv. pruni).",
        ["Angular dark water-soaked leaf spots that drop out leaving ragged 'shot-holes'.", "Yellowing and early leaf drop."],
        ["Plant resistant peach cultivars and establish windbreaks on exposed sandy sites.", "Avoid excessive nitrogen."],
        ["Consult local stone-fruit extension guidance for dormant and seasonal orchard management."],
    ),
    "peach__healthy": _healthy_info("Peach"),
    "pepper_bell__bacterial_spot": _info(
        "Bacterial Spot", "Pepper (Bell)", False,
        "Bacterial disease caused by Xanthomonas spp. spread rapidly by rain splash and overhead irrigation.",
        ["Small irregular water-soaked spots turning dark brown/black on leaves.", "Yellowing halos and premature leaf drop."],
        ["Use certified disease-free seed and resistant varieties.", "Rotate crops 2–3 years and use drip irrigation with mulch."],
        ["Remove severely affected leaves/plants early and avoid working in wet foliage.", "Consult local extension guidance for approved preventive management."],
    ),
    "pepper_bell__healthy": _healthy_info("Pepper (Bell)"),
    "potato__early_blight": _info(
        "Early Blight", "Potato", False,
        "Fungal foliar disease (Alternaria solani) forming concentric target-like spots on older leaves.",
        ["Dark brown circular spots with concentric rings ('bull's-eye' pattern).", "Yellowing around lesions starting on lower foliage."],
        ["Rotate away from solanaceous crops for 3 years.", "Use certified seed tubers, steady watering, and balanced fertility."],
        ["Remove severely blighted lower foliage on small plots.", "Consult local agricultural extension guidance for protective disease management."],
    ),
    "potato__healthy": _healthy_info("Potato"),
    "potato__late_blight": _info(
        "Late Blight", "Potato", False,
        "Fast-spreading oomycete disease (Phytophthora infestans) thriving in cool, wet weather.",
        ["Large irregular water-soaked dark blotches on leaf tips and margins.", "White fuzzy sporulation on leaf undersides under high humidity."],
        ["Plant certified blight-free seed tubers and destroy volunteer potatoes/cull piles.", "Hill soil over tubers and monitor regional blight alerts."],
        ["Immediately isolate and dispose of confirmed infected plants in bags (do not compost).", "Seek urgent local agricultural extension guidance."],
    ),
    "raspberry__healthy": _healthy_info("Raspberry"),
    "soybean__healthy": _healthy_info("Soybean"),
    "squash__powdery_mildew": _info(
        "Powdery Mildew", "Squash", False,
        "Common cucurbit fungal disease producing talcum-like white coatings on older leaves.",
        ["White powdery patches expanding across upper and lower leaf surfaces.", "Yellowing, drying, and collapse of heavily coated leaves."],
        ["Plant mildew-resistant squash varieties in full sun with generous spacing.", "Clear crop residue after harvest."],
        ["Prune heavily affected senescing leaves and consult local gardening/extension guidance for approved preventive controls."],
    ),
    "strawberry__healthy": _healthy_info("Strawberry"),
    "strawberry__leaf_scorch": _info(
        "Leaf Scorch", "Strawberry", False,
        "Fungal disease (Diplocarpon earlianum) causing coalescing purplish blotches and scorched foliage.",
        ["Irregular dark purple-to-reddish blotches on upper leaf surfaces.", "Leaves dry, curl upward, and appear scorched in severe cases."],
        ["Plant resistant cultivars, use drip irrigation with straw mulch, and renovate beds after harvest."],
        ["Remove heavily scorched foliage and consult local berry extension guides for seasonal management."],
    ),
    "tomato__bacterial_spot": _info(
        "Bacterial Spot", "Tomato", False,
        "Bacterial leaf and fruit disease caused by Xanthomonas spp. spread via splash and handling.",
        ["Small dark water-soaked circular/angular spots on leaves.", "Yellowing around clustered spots and lower defoliation."],
        ["Use certified disease-free seed, mulch soil, water at base, and rotate crops 2–3 years.", "Avoid touching wet plants."],
        ["Prune and discard heavily affected lower leaves away from beds.", "Consult local agricultural extension guidance for integrated control."],
    ),
    "tomato__early_blight": _info(
        "Early Blight", "Tomato", False,
        "Common fungal disease caused by Alternaria solani beginning on older lower tomato leaves.",
        ["Brown-to-black circular spots showing concentric target rings.", "Yellowing around infected areas and progressive lower leaf loss."],
        ["Stake/prune lower leaves near soil, apply mulch, avoid overhead watering, and rotate beds."],
        ["Remove severely infected lower leaves promptly and improve air circulation.", "Consult local agricultural guidance for approved protective options."],
    ),
    "tomato__healthy": _healthy_info("Tomato"),
    "tomato__late_blight": _info(
        "Late Blight", "Tomato", False,
        "Aggressive oomycete disease (Phytophthora infestans) capable of rapid foliage and fruit collapse.",
        ["Large irregular water-soaked greenish-black leaf blotches and stem lesions.", "Fuzzy white sporulation on damp leaf undersides."],
        ["Choose resistant cultivars, eliminate nearby volunteer potatoes/tomatoes, and keep foliage dry."],
        ["Bag and remove confirmed infected plants immediately to limit airborne spread; contact local extension services."],
    ),
    "tomato__leaf_mold": _info(
        "Leaf Mold", "Tomato", False,
        "Fungal disease (Passalora fulva) prevalent in humid greenhouses and dense plantings.",
        ["Pale green to yellow patches on upper leaf surfaces.", "Olive-green to purplish velvety mold directly beneath patches on leaf undersides."],
        ["Improve greenhouse ventilation to keep humidity below 85%.", "Water at soil level and select resistant varieties."],
        ["Carefully remove affected lower leaves and sanitize structures between seasons; consult extension guidance."],
    ),
    "tomato__septoria_leaf_spot": _info(
        "Septoria Leaf Spot", "Tomato", False,
        "Fungal foliage disease (Septoria lycopersici) causing abundant small circular gray-centered spots.",
        ["Numerous tiny circular spots with dark borders and pale gray/tan centers.", "Rapid yellowing and defoliation progressing upward from lower leaves."],
        ["Clear all tomato debris after harvest, disinfect stakes, apply mulch, and avoid overhead irrigation."],
        ["Prune affected lower leaves early and consult local extension recommendations for preventive care."],
    ),
    "tomato__spider_mites_two_spotted": _info(
        "Spider Mites (Two-Spotted Spider Mite)", "Tomato", False,
        "Sap-feeding arthropod pest (Tetranychus urticae) thriving in hot, dry, dusty conditions.",
        ["Fine yellow/white pinpoint stippling across upper leaf surfaces.", "Bronzed brittle leaves and fine webbing underneath foliage."],
        ["Avoid drought stress, minimize surrounding dust, and conserve beneficial predatory insects."],
        ["Inspect leaf undersides closely and consult local integrated pest management (IPM) extension guidance."],
    ),
    "tomato__target_spot": _info(
        "Target Spot", "Tomato", False,
        "Fungal disease (Corynespora cassiicola) producing sunken concentric brown lesions.",
        ["Pinpoint spots expanding into sunken brown circular lesions with concentric rings and yellow halos.", "Progressive lower-canopy defoliation."],
        ["Maximize airflow via staking/spacing, use drip irrigation + mulch, and rotate crops."],
        ["Remove heavily affected lower leaves and consult local extension guidance for management schedules."],
    ),
    "tomato__tomato_mosaic_virus": _info(
        "Tomato Mosaic Virus", "Tomato", False,
        "Mechanically transmitted tobamovirus spread easily via hands, tools, and plant contact.",
        ["Light-and-dark green mosaic mottling across leaflets.", "Narrowed, twisted, or blistered young foliage."],
        ["Use resistant cultivars and certified seed; wash hands and disinfect tools thoroughly before handling."],
        ["No antiviral cure exists; carefully bag and remove symptomatic plants without brushing healthy foliage and follow extension sanitation protocols."],
    ),
    "tomato__tomato_yellow_leaf_curl_virus": _info(
        "Tomato Yellow Leaf Curl Virus", "Tomato", False,
        "Whitefly-vectored begomovirus causing severe leaf curling, yellowing, and stunting.",
        ["Strong upward cupping/curling of young leaves with yellow margins.", "Stunted bushy growth and heavy blossom drop."],
        ["Grow TYLCV-resistant varieties, use insect-exclusion mesh in nurseries, and monitor whitefly vectors."],
        ["No chemical cure exists for viral infection; promptly rogue confirmed infected plants and consult local extension IPM guidance for whitefly management."],
    ),
}


class DiseaseInformationService:
    """Retrieves structured educational disease guidance for verified canonical classes."""

    def get_information(
        self,
        canonical_class: str,
        plant_name: str,
        disease_name: str,
        is_healthy: bool,
    ) -> Optional[Dict[str, Any]]:
        """Look up structured educational metadata; falls back cleanly for unknown classes."""
        if not canonical_class:
            return None

        entry = _KNOWLEDGE_BASE.get(canonical_class.strip().lower())
        if entry is not None:
            return dict(entry)

        if is_healthy:
            return _healthy_info(plant_name)

        return _info(
            disease_name=disease_name,
            affected_plant=plant_name,
            is_healthy=False,
            summary=(
                f"General educational overview for '{disease_name}' affecting {plant_name}. "
                "Detailed local profile is not yet cataloged in the knowledge base."
            ),
            symptoms=[
                "Visible foliar discoloration, spotting, or structural stress patterns on the leaf.",
            ],
            prevention=[
                "Avoid overhead watering, maintain good airflow, and practice crop rotation and sanitation.",
            ],
            treatment=[
                "Isolate severely affected foliage where feasible and consult a local agricultural extension specialist for confirmed diagnosis.",
            ],
        )
