
from RagAdaptation.core.model_config import ModelConfig
from RagAdaptation.evaluate_questions import generate_answer
from RagAdaptation.compute_probs_updated import compute_probs
query="Are Chrysalis and Look both women's magazines?"

model, tok, device = ModelConfig("microsoft/Phi-3-mini-4k-instruct").load()
mid="microsoft/Phi-3-mini-4k-instruct"
prompt = ModelConfig("microsoft/Phi-3-mini-4k-instruct").format_prompt(
                question=query,
                context="",
                context_cite_at2_formating=False,
            )

print(prompt)
generated_answer = generate_answer(model, tok, prompt, max_new_tokens=50)
print(f"\n generated answer: {generated_answer}")


stats_list, full_logps_without_context = compute_probs(model=model, tok=tok, prompts=[prompt], device=model.device,
                                                                   expected_result=None, batch_size=1,
                                                                   masked_context_list=None,
                                                                   true_variants=ModelConfig("microsoft/Phi-3-mini-4k-instruct").true_variants,
                                                                   false_variants=ModelConfig("microsoft/Phi-3-mini-4k-instruct").false_variants,
                                                                   return_full_logp=False,
                                                                   file_name=f"compute_probs_baseline_without_context_{mid.replace('/', '__')}_idx{3}.txt",
                                                                   detect_flip_to_true=("false" == "false"),save_file=False,stop_on_flip=True )
