"""Local standard-library HTTP server for the Yeast Digital Twin handover app."""

from __future__ import annotations

import csv
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .core import (build_culture_manifest, infer_column_roles, load_dataset, predict_trajectory, predict_with_observations,
                   run_metadata, screen_candidates, split_cultures, train_candidate_models,
                   validate_dataset, verify_with_yeast9)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STATIC = ROOT / "static"
DEMO = ROOT / "demo" / "active_rxncon_observable_clean.csv"
SESSIONS = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "YDTHandOver/1.0"

    def _send(self, status=200, payload=None, content_type="application/json", headers=None):
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload).encode()
        elif isinstance(payload, str): body = payload.encode()
        else: body = payload or b""
        self.send_response(status); self.send_header("Content-Type", content_type); self.send_header("Content-Length", str(len(body)))
        for key,value in (headers or {}).items(): self.send_header(key,value)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/health": return self._send(payload={"ok":True,"service":"Yeast Digital Twin handover API"})
        if path=="/api/example":
            return self._send(payload=DEMO.read_text(), content_type="text/csv", headers={"Content-Disposition":"attachment; filename=yeast_digital_twin_demo.csv"})
        if path=="/methodology":
            report=REPO_ROOT / "YEAST_DIGITAL_TWIN_COMPREHENSIVE_TECHNICAL_WRITEUP.md"
            if report.exists(): return self._send(payload=report.read_bytes(), content_type="text/markdown; charset=utf-8")
            return self._send(404,{"error":"The technical writeup is not available."})
        if path.startswith("/api/export/"):
            return self.export(path.split("/")[-1], parse_qs(urlparse(self.path).query).get("run_id", [None])[0])
        rel="index.html" if path in ("/","") else path.removeprefix("/")
        file=STATIC / rel
        if file.exists() and file.is_file(): return self._send(payload=file.read_bytes(),content_type=mimetypes.guess_type(str(file))[0] or "application/octet-stream")
        return self._send(404,{"error":"Not found"})

    def _json(self):
        n=int(self.headers.get("Content-Length","0")); return json.loads(self.rfile.read(n).decode())

    def do_POST(self):
        path=urlparse(self.path).path
        try:
            data=self._json()
            if path=="/api/analyze": return self.analyze(data)
            if path=="/api/validate": return self.validate(data)
            if path=="/api/train": return self.train(data)
            if path=="/api/predict": return self.predict(data)
            if path=="/api/update": return self.update(data)
            if path=="/api/load_model": return self.load_model(data)
            if path=="/api/screen": return self.screen(data)
            if path=="/api/verify": return self._send(payload=verify_with_yeast9())
            return self._send(404,{"error":"Not found"})
        except ValueError as exc: return self._send(400,{"error":str(exc)})
        except Exception as exc: return self._send(500,{"error":f"The operation could not be completed: {exc}"})

    def analyze(self,data):
        df,digest=load_dataset(data.get("csv_text",""),data.get("filename","upload.csv")); suggestions=infer_column_roles(df)
        mapping={"culture_id":[],"time":[],"environment":[],"product":[],"biomass":[],"reporter":[],"genotype":[],"ignore":[]}
        for col,role in suggestions.items(): mapping[role].append(col)
        session_id=digest[:12]; SESSIONS[session_id]={"df":df,"hash":digest,"mapping":mapping,"filename":data.get("filename","upload.csv")}
        return self._send(payload={"run_id":session_id,"hash":digest,"headers":list(df.columns),"suggestions":suggestions,"mapping":mapping,"preview":df.head(8).fillna("").to_dict(orient="records"),"validation":validate_dataset(df,mapping)})

    def session(self,data):
        sid=data.get("run_id");
        if sid not in SESSIONS: raise ValueError("This upload session has expired. Upload the dataset again.")
        return SESSIONS[sid]

    def validate(self,data):
        s=self.session(data); s["mapping"]=data.get("mapping",s["mapping"]); result=validate_dataset(s["df"],s["mapping"]); return self._send(payload={"validation":result,"mapping":s["mapping"]})

    def train(self,data):
        s=self.session(data); s["mapping"]=data.get("mapping",s["mapping"]); validation=validate_dataset(s["df"],s["mapping"])
        if not validation["ok"]: raise ValueError("Fix the dataset errors before training.")
        manifest=build_culture_manifest(s["df"],s["mapping"]); splits=split_cultures(manifest,seed=int(data.get("seed",42))); training=train_candidate_models(s["df"],s["mapping"],splits,data.get("config",{})); training["mapping"]=s["mapping"]; s["training"]=training; s["splits"]=splits; s["mapping"]=data["mapping"]
        return self._send(payload={"training":training,"manifest":manifest,"metadata":run_metadata(s["hash"],s["mapping"],splits,training)})

    def predict(self,data):
        s=self.session(data)
        if "training" not in s: raise ValueError("Train or load a model before predicting.")
        return self._send(payload=predict_trajectory(s["training"]["best"],data.get("condition",{})))

    def update(self,data):
        s=self.session(data)
        if "training" not in s: raise ValueError("Train or load a model before updating a prediction.")
        observations=data.get("observations",{})
        if data.get("partial_csv"):
            partial,_=load_dataset(data["partial_csv"],"ongoing_culture.csv"); roles=s.get("mapping",{}); cid=roles.get("culture_id",[partial.columns[0]])[0]; time=roles.get("time",[partial.columns[1]])[0]
            group=partial.sort_values(time)
            for col in s["training"]["best"]["model"].get("observation_columns",[]):
                if col in group: observations[col]=pd.to_numeric(group[col],errors="coerce").tolist()
            observations["cutoff_fraction"]=min(0.99,len(group)/max(len(s["training"]["best"]["model"].get("target_grid",[])),1))
        return self._send(payload=predict_with_observations(s["training"]["best"],data.get("condition",{}),observations))

    def load_model(self,data):
        training=data.get("training")
        if not training or "best" not in training: raise ValueError("This file is not a valid Yeast Digital Twin model run.")
        sid=data.get("run_id",training.get("run_id","loaded")); SESSIONS[sid]={"training":training,"mapping":data.get("mapping",{}),"hash":data.get("dataset_hash","loaded")}
        return self._send(payload={"run_id":sid,"training":training,"message":"Model loaded. Dataset is not required for prediction."})

    def screen(self,data):
        s=self.session(data)
        if "training" not in s: raise ValueError("Train or load a model before screening.")
        rows=screen_candidates(s["training"]["best"],data.get("ranges",{}),int(data.get("n_candidates",100))); s["screen"]=rows
        return self._send(payload={"rows":rows,"verification":verify_with_yeast9()})

    def export(self,kind,sid=None):
        sid=sid or self.headers.get("X-YDT-Run") or next(iter(SESSIONS),None)
        s=SESSIONS.get(sid)
        if not s: return self._send(404,{"error":"No active run."})
        if kind=="dataset": body=s["df"].to_csv(index=False); name="mapped_dataset.csv"
        elif kind=="mapping": body=json.dumps(s["mapping"],indent=2); name="column_mapping.json"
        elif kind=="model" and "training" in s: body=json.dumps(s["training"],indent=2); name="model_run.json"
        elif kind=="screen" and "screen" in s:
            keys=["rank","candidate_id","final_product","auc","verification"]
            body=csv.DictWriter(__import__("io").StringIO(),fieldnames=keys)
            stream=body # handled below
            out=__import__("io").StringIO(); w=csv.DictWriter(out,fieldnames=keys); w.writeheader();
            for row in s["screen"]: w.writerow({k:row.get(k) for k in keys})
            body=out.getvalue(); name="candidate_ranking.csv"
        else: return self._send(404,{"error":"That export is not available yet."})
        return self._send(payload=body,content_type="text/plain",headers={"Content-Disposition":f"attachment; filename={name}"})


def main():
    import argparse
    p=argparse.ArgumentParser(description="Run the Yeast Digital Twin handover app")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=8765); args=p.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),Handler); print(f"Yeast Digital Twin app: http://{args.host}:{args.port}",flush=True); server.serve_forever()


if __name__ == "__main__": main()
