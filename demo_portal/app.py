from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from .auth import current_role, current_user, is_authenticated, login_required, role_required
from .db import authenticate_user, init_db, list_seed_accounts
from .repository import (
    OUTPUTS,
    ROOT,
    get_audit_bundle,
    get_comparison,
    get_fashion_bundle,
    get_federation_bundle,
    get_live_monitor,
    get_overview,
    get_paper_bundle,
    get_paper_table_bundle,
    get_reference_file,
    get_research_bundle,
    get_run,
    list_comparisons,
    list_runs,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".svg"}


def _resolve_artifact_path(relpath: str) -> Path:
    resolved = (ROOT / relpath).resolve()
    root_resolved = ROOT.resolve()
    if root_resolved not in resolved.parents and resolved != root_resolved:
        abort(403)
    if not resolved.exists():
        abort(404)
    return resolved


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["SECRET_KEY"] = os.environ.get("PAPER_DEMO_SECRET", "fed-diff-demo-secret")
    init_db()

    @app.context_processor
    def inject_globals():
        return {
            "is_authenticated": is_authenticated(),
            "current_role": current_role(),
            "current_user": current_user(),
        }

    @app.route("/")
    def index():
        if is_authenticated():
            return redirect(url_for("dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = None
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = authenticate_user(username, password)
            if user is not None:
                session["authenticated"] = True
                session["username"] = user["username"]
                session["display_name"] = user["display_name"]
                session["role"] = user["role"]
                next_path = request.args.get("next") or url_for("dashboard")
                return redirect(next_path)
            error = "用户名或密码不正确。"
        return render_template("login.html", error=error, demo_accounts=list_seed_accounts())

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        session.clear()
        return redirect(url_for("login"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        return render_template("dashboard.html", overview=get_overview(), live_monitor=get_live_monitor())

    @app.route("/experiments")
    @login_required
    def experiments():
        return render_template("experiments.html", runs=list_runs())

    @app.route("/results")
    @login_required
    def results():
        runs = list_runs()
        datasets = list(dict.fromkeys(run.get("dataset", "unknown") for run in runs))
        comparisons = list_comparisons()
        return render_template("results.html", runs=runs, datasets=datasets, comparisons=comparisons)

    @app.route("/sampling-lab")
    @login_required
    def sampling_lab():
        runs = list_runs()
        federation_bundle = get_federation_bundle()
        dataset_groups: dict[str, list[dict]] = {}
        for run in runs:
            dataset = run.get("dataset", "unknown")
            dataset_groups.setdefault(dataset, []).append(run)
        return render_template(
            "sampling_lab.html",
            runs=runs,
            dataset_groups=dataset_groups,
            comparison_rows=federation_bundle.get("comparison_rows", []),
        )

    @app.route("/fashion-spotlight")
    @login_required
    def fashion_spotlight():
        return render_template("fashion_spotlight.html", fashion_bundle=get_fashion_bundle())

    @app.route("/architecture")
    @login_required
    def architecture():
        return render_template("architecture.html", runs=list_runs(), overview=get_overview())

    @app.route("/research")
    @login_required
    def research():
        return render_template("research.html", research_bundle=get_research_bundle())

    @app.route("/comparisons/<comparison_id>")
    @login_required
    def comparison_detail(comparison_id: str):
        comparison = get_comparison(comparison_id)
        if comparison is None:
            abort(404)
        return render_template("comparison_detail.html", comparison=comparison)

    @app.route("/experiments/<run_id>")
    @role_required("admin")
    def experiment_detail(run_id: str):
        run = get_run(run_id)
        if run is None:
            abort(404)
        return render_template("experiment_detail.html", run=run)

    @app.route("/monitor")
    @role_required("admin")
    def monitor():
        return render_template("monitor.html", live_monitor=get_live_monitor())

    @app.route("/audit")
    @role_required("admin")
    def audit():
        return render_template("audit.html", audit_bundle=get_audit_bundle())

    @app.route("/federation")
    @login_required
    def federation():
        return render_template("federation.html", federation_bundle=get_federation_bundle())

    @app.route("/paper")
    @role_required("admin")
    def paper():
        return render_template("paper.html", paper_bundle=get_paper_bundle())

    @app.route("/paper-tables")
    @role_required("admin")
    def paper_tables():
        return render_template("paper_tables.html", paper_table_bundle=get_paper_table_bundle())

    @app.route("/reference-files/<doc_id>")
    @login_required
    def reference_file(doc_id: str):
        file_meta = get_reference_file(doc_id)
        if file_meta is None:
            abort(404)
        return send_file(file_meta["path"])

    @app.route("/artifacts/<path:relpath>")
    @role_required("admin")
    def artifact(relpath: str):
        return send_file(_resolve_artifact_path(relpath))

    @app.route("/preview-artifacts/<path:relpath>")
    @login_required
    def preview_artifact(relpath: str):
        resolved = _resolve_artifact_path(relpath)
        if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
            abort(403)
        outputs_resolved = OUTPUTS.resolve()
        if outputs_resolved not in resolved.parents and resolved != outputs_resolved:
            abort(403)
        return send_file(resolved)

    @app.route("/api/overview")
    @login_required
    def api_overview():
        return jsonify(get_overview())

    @app.route("/api/monitor/live")
    @role_required("admin")
    def api_monitor():
        return jsonify(get_live_monitor())

    @app.route("/api/runs")
    @login_required
    def api_runs():
        return jsonify(list_runs())

    @app.route("/api/runs/<run_id>")
    @role_required("admin")
    def api_run_detail(run_id: str):
        run = get_run(run_id)
        if run is None:
            abort(404)
        return jsonify(run)

    @app.errorhandler(403)
    def forbidden(_error):
        return render_template(
            "status.html",
            title="没有权限访问",
            message="当前账号没有权限查看这个模块。管理员账号可以查看完整样图、训练监控、安全审计和论文资料页。",
        ), 403

    @app.errorhandler(404)
    def not_found(_error):
        return render_template(
            "status.html",
            title="页面不存在",
            message="你访问的页面或资源不存在，可能是路径错误，或者对应实验结果尚未生成。",
        ), 404

    return app
