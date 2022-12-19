# Streamlit demo app

This demo app will expose web UI to interactively explain models.

## Run

To run this app You have to be in root directory of this repository. To
launch Streamlit app just type:
```bash
poetry run streamlit run example/streamlit_app/run_streamlit.py --server.port 8080
```

## Log directory

Application is fetching experiment data inside path given in environment
variables with key `LOGDIR`. You can pass custom path.

The catalog has a specific structure. The first level is the date of the
experiment. The second level is the UUID of the experiment performed on
that day. Inside it are already contained the data needed for the explanation:
the data directory with sample data accessible from the explanation
application, the labels directory with the `idx_to_label.json.pkl` file,
which contains the JSON with the index-class mapping, and the training
directory, which contains directories corresponding to the training epoch
number, in which the models are stored, always with the same name
`model.onnx` in ONNX format.

Example log directory structure:
```bash
logs/
└── 2022-12-13
    └── 777582e2-7af8-11ed-98c0-91481eedfd34
        ├── data
        |   └── data.pkl
        ├── labels
        |   └── idx_to_label.json.pkl
        └── training
            └── 0
                └── model.onnx

```
