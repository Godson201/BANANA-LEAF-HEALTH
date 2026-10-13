# LeafLens — HTML frontend and Flask backend

Start `Start_Dashboard.cmd` from the project root, then open http://127.0.0.1:5050. Opening `index.html` directly cannot call the model backend.

- `index.html`, `static/style.css`, `static/app.js`: frontend.
- `app.py`: saved classifier, upload validation and prediction routes.
- `leaf_screening.py`: heuristic banana-photo subject screen.
- `assets/`: labeled examples and evaluation charts.

Install the root notebook dependencies and `dashboard/requirements.txt`. Prepare screening weights with `tools/prepare_leaf_screening.py` if needed. Predictions are withheld when screening is unavailable, except for explicit user confirmation. Scores are uncalibrated and may be wrong. Known sample buttons demonstrate the original banana classifier directly. Uploaded images are not retained.

Run backend checks with `python -m unittest discover -s dashboard/tests -v` from the project root.

The model selector offers the original four-class model and a separate six-class research preview. The latter adds Fusarium/Panama and Moko, but not Xanthomonas wilt or bunchy top. Its 98.01% test accuracy is based on a small mixed-source test set and is not field validation. See `MODEL_EXPANSION.md` in the project root.

Farmer next steps are linked to their information sources. The whole-plant checklist is explicitly rule-based. It does not infer a new disease or prescribe pesticide rates. Result downloads can include an optional field note and the latest checklist result. Nothing is uploaded to an external service by the app.

Frontend integration checks: install Node.js, run `npm install --prefix dashboard/tests`, then `npm test --prefix dashboard/tests` while the server is running. Dependencies under node_modules are not included in submission archives.
