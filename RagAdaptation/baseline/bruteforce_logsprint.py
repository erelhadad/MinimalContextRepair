# from __future__ import annotations
# import argparse
# import multiprocessing as mp
# import time
# from pathlib import Path
# import sys
#
#
# _THIS_FILE = Path(__file__).resolve()
# _PKG_DIR = _THIS_FILE.parents[1]          # .../RagAdaptation
# _PROJECT_ROOT = _PKG_DIR.parent           # .../RAG_EXP
# if str(_PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(_PROJECT_ROOT))
#
# from RagAdaptation.compute_probs_updated import compute_probs
# from RagAdaptation.core.documents import combine_document_text, load_documents_any
# from RagAdaptation.core.models import get_hf_scorer
# from RagAdaptation.core.artifacts import ensure_dir, write_text
# from RagAdaptation.prompts_format import TF_RAG_TEMPLATE
# from RagAdaptation.baseline.bruteforce_common import (
#     TIME_LIMIT_SECONDS,
#     bruteforce_output_dir,
#     create_masked_prompts,
#     plot_histograms,
#     resolve_document_path,
#     tokenize_context_with_offsets,
# )
#
#
# def _run_exp_wrapper(query, k, batch_size, hf_model_id, pdf_path, out_dir):
#     run_exp(query=query, k=k, batch_size=batch_size, hf_model_id=hf_model_id, pdf_path=pdf_path, out_dir=out_dir)
#
#
# def run_exp(query: str, k: int, batch_size: int, hf_model_id: str, pdf_path: str | Path, out_dir: str | Path):
#     document = load_documents_any(pdf_path)
#     full_context = combine_document_text(document)
#     hf_model, hf_tok, hf_device = get_hf_scorer(hf_model_id)
#
#     _, offsets = tokenize_context_with_offsets(full_context, hf_tok)
#     print("Tokens amount in total:", len(offsets))
#
#     masked_prompts, masked_context_list = create_masked_prompts(full_context, query, offsets, k=k)
#
#     baseline_prompt = TF_RAG_TEMPLATE.format(context=full_context, question=query)
#     baseline_stats, _ = compute_probs(
#         hf_model, hf_tok, [baseline_prompt], hf_device, None,
#         batch_size=1, return_full_logp=True,
#         file_name=str(Path(out_dir) / "baseline_compute_probs.txt"),
#     )
#     baseline_stats = baseline_stats[0]
#     print("BASELINE:", baseline_stats)
#
#     masked_stats, masked_logps = compute_probs(
#         hf_model, hf_tok, masked_prompts, hf_device, None, batch_size,
#         masked_context_list, return_full_logp=True,
#         file_name=str(Path(out_dir) / "masked_compute_probs.txt"),
#     )
#
#     print("Masked prompts scored:", len(masked_stats))
#     plot_histograms(masked_logps, out_dir=Path(out_dir) / "plots", tag="masked_prompts")
#
#
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--query", type=str, default="Is sugar considered an addictive substance?")
#     parser.add_argument("--k", type=int, default=2)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--hf_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
#     parser.add_argument("--document", type=str, default="sugar")
#     parser.add_argument("--out_dir", type=str, default="")
#     args = parser.parse_args()
#
#     pdf_path = resolve_document_path(args.document)
#     out_dir = Path(args.out_dir) if args.out_dir else bruteforce_output_dir(Path(pdf_path).stem)
#     ensure_dir(out_dir)
#
#     p = mp.Process(
#         target=_run_exp_wrapper,
#         args=(args.query, args.k, args.batch_size, args.hf_model_id, pdf_path, out_dir),
#         daemon=True,
#     )
#
#     start = time.perf_counter()
#     p.start()
#     p.join(TIME_LIMIT_SECONDS)
#
#     if p.is_alive():
#         msg = f"[TIMEOUT] Exceeded {TIME_LIMIT_SECONDS} seconds. Terminating experiment..."
#         p.terminate()
#         p.join(30)
#         if p.is_alive():
#             msg += "\n[TIMEOUT] Still alive after terminate(); killing."
#             p.kill()
#             p.join()
#     else:
#         elapsed = time.perf_counter() - start
#         msg = f"[DONE] Finished in {elapsed:.2f} seconds."
#
#     write_text(Path(out_dir) / "runtime_status.txt", msg)
#     print(msg)
#
#
# if __name__ == "__main__":
#     main()
