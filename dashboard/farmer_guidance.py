"""Reviewed, source-linked guidance. This is decision support, not diagnosis."""
SOURCES = {
    'cordana': {'title': 'ProMusa: Cordana leaf spot', 'url': 'https://www.promusa.org/Cordana+leaf+spot'},
    'sigatoka': {'title': 'ProMusa: Sigatoka leaf spot', 'url': 'https://www.promusa.org/Sigatoka+leaf+spot'},
    'fusarium': {'title': 'FAO: preventing the spread of Fusarium TR4', 'url': 'https://www.fao.org/world-banana-forum/fusariumtr4/tr4-global-network/actions/en/'},
    'bunchy': {'title': 'IITA: banana bunchy top management', 'url': 'https://blogs.iita.org/empowering-farmers-to-fight-banana-bunchy-top-disease-through-knowledge-sharing/'},
    'bacterial': {'title': 'IITA: Xanthomonas wilt management in Rwanda', 'url': 'https://iita.org/news-item/novel-and-effective-technique-to-control-banana-xanthomonas-wilt-is-suitable-for-rwandan-agroecology/'},
}
CATALOG = [
    {'id': 'healthy', 'name': 'Healthy appearance', 'category': 'Routine monitoring', 'model_supported': True,
     'summary': 'The photo resembles the healthy training class. This does not rule out hidden infection, pests or nutrient problems.',
     'signs': 'A healthy-looking leaf can occur on a plant with problems elsewhere.',
     'actions': ['Inspect the entire plant and neighboring plants regularly.', 'Use reliable, healthy planting material and maintain farm hygiene.', 'Keep dated photos so changes are easier to discuss with an extension officer.'], 'source': SOURCES['fusarium']},
    {'id': 'cordana', 'name': 'Cordana leaf spot', 'category': 'Leaf-spot guidance', 'model_supported': True,
     'summary': 'A leaf-spot pattern that may also occur alongside other leaf damage.',
     'signs': 'Brown oval patches with a darker edge and yellow border can occur. Similar spots have other causes.',
     'actions': ['Compare several leaves and photograph the affected areas.', 'Check for other stress or disease rather than relying on one spot pattern.', 'Ask an extension officer for advice if damage spreads or healthy leaf area declines.'], 'source': SOURCES['cordana']},
    {'id': 'pestalotiopsis', 'name': 'Pestalotiopsis class', 'category': 'Confirm the cause', 'model_supported': True,
     'summary': 'This is a dataset label, not laboratory identification of a fungus.',
     'signs': 'Leaf spotting and damaged tissue are not specific enough to confirm the pathogen from one photo.',
     'actions': ['Record when spotting began and whether neighboring plants are affected.', 'Take an additional whole-plant photo and a sharp close-up.', 'Request local plant-health advice before choosing a treatment.'], 'source': SOURCES['cordana']},
    {'id': 'sigatoka', 'name': 'Sigatoka-like leaf spots', 'category': 'Protect healthy leaf area', 'model_supported': True,
     'summary': 'The model has a general Sigatoka label; it does not distinguish black from yellow Sigatoka.',
     'signs': 'Streaks and expanding dead patches may reduce functional leaf area.',
     'actions': ['Review drainage, weed control, nutrition and plant spacing with local advice.', 'Ask an extension officer how to manage affected leaf tissue while preserving healthy leaves.', 'If treatment is recommended, use only locally approved products and follow their labels.'], 'source': SOURCES['sigatoka']},
    {'id': 'fusarium_wilt', 'name': 'Fusarium wilt / Panama disease', 'category': 'Extension review needed', 'model_supported': False,
     'summary': 'Whole-plant yellowing or wilting needs expert assessment. A leaf photograph cannot establish Fusarium race or confirm TR4.',
     'signs': 'Yellowing and wilt can resemble other diseases and environmental stress.',
     'actions': ['Avoid moving soil, suckers and equipment from a suspect area.', 'Use certified healthy planting material and follow local farm biosecurity instructions.', 'Contact an extension officer or plant-protection service for confirmation and outbreak advice.'], 'source': SOURCES['fusarium']},
    {'id': 'bunchy_top', 'name': 'Banana bunchy top', 'category': 'Extension review needed', 'model_supported': False,
     'summary': 'Inspect the entire growing point. This disease is not a prediction class in the current model.',
     'signs': 'A bunched, stunted growing point is a reason to seek specialist advice, not confirmation of a virus.',
     'actions': ['Photograph the whole plant, growing point and affected leaves.', 'Do not exchange suckers from a suspect plant with another farm.', 'Ask an extension officer about clean planting material, aphid management and confirmed-disease procedures.'], 'source': SOURCES['bunchy']},
    {'id': 'bacterial_wilt', 'name': 'Bacterial wilt / Xanthomonas wilt', 'category': 'Extension review needed', 'model_supported': False,
     'summary': 'Several diseases cause bacterial wilt. Moko and Xanthomonas wilt are different; this guide focuses on Xanthomonas wilt.',
     'signs': 'Whole-plant wilt and changes in flowers or fruit should be examined by a trained adviser.',
     'actions': ['Avoid sharing cutting tools between suspect and healthy plants.', 'Ask a local adviser to demonstrate tool sanitation and safe field practices.', 'For confirmed Xanthomonas wilt, discuss recommended diseased-stem management; do not remove plants based only on this app.'], 'source': SOURCES['bacterial']},
]
BY_ID = {item['id']: item for item in CATALOG}
BY_ID['fusarium_wilt']['experimental_available'] = True
BY_ID['pestalotiopsis']['source'] = {'title': 'General banana biosecurity guidance (FAO)', 'url': SOURCES['fusarium']['url']}
CATALOG.append({'id': 'moko_wilt', 'name': 'Moko bacterial wilt', 'category': 'Experimental model class',
    'model_supported': False, 'experimental_available': True,
    'summary': 'Available only in the expanded research model. This is not the same disease as Xanthomonas wilt.',
    'signs': 'Wilting and fruit changes require a trained plant-health assessment; a model score cannot confirm the bacterium.',
    'actions': ['Ask an extension officer to assess the whole plant.', 'Avoid exchanging planting material from a suspect plant.', 'Follow local plant-protection instructions before removing or treating plants.'],
    'source': {'title': 'Australian agriculture department: blood and Moko diseases', 'url': 'https://www.agriculture.gov.au/biosecurity-trade/pests-diseases-weeds/plant/identify/blood-and-moko-diseases-banana'}})
BY_ID['moko_wilt'] = CATALOG[-1]
SYMPTOMS = {'whole_plant_wilt', 'bunched_growth', 'flower_fruit_changes', 'spreading', 'leaf_spots'}

def care_for(class_name, tentative=False):
    care = dict(BY_ID.get(class_name, BY_ID['healthy']))
    care['basis'] = 'General guidance for the predicted class; not an individually confirmed treatment plan.'
    care['review_note'] = 'The prediction is tentative. Confirm the cause before acting.' if tentative else 'Compare the result with the whole plant and seek advice if symptoms persist.'
    return care

def field_check(symptoms):
    selected = set(symptoms) & SYMPTOMS
    priority = bool(selected & {'whole_plant_wilt', 'bunched_growth', 'flower_fruit_changes', 'spreading'})
    return {'basis': 'Rule-based check of farmer-reported symptoms; not an AI disease diagnosis.',
        'priority': 'extension_review' if priority else 'monitor',
        'title': 'Arrange an extension review' if priority else 'Record and monitor the plant',
        'message': 'These symptoms need a whole-plant assessment, even if the leaf model says Healthy.' if priority else 'Keep clear, dated photos. Seek advice if symptoms persist or spread.',
        'actions': ['Photograph the whole plant and affected areas.', 'Note when symptoms began and how many plants are affected.', 'Avoid moving planting material from suspect plants.'],
        'related_guides': [name for flag, name in [('whole_plant_wilt', 'fusarium_wilt'), ('bunched_growth', 'bunchy_top'), ('flower_fruit_changes', 'bacterial_wilt')] if flag in selected]}
