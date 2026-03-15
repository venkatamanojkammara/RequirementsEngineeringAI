# -----------------------------------------------------------------
# Imports
# -----------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# -----------------------------------------------------------------
# Model Configuration
# -----------------------------------------------------------------
BASE_MODEL = "microsoft/phi-2"
ADAPTER_PATH = "./model/requirement_engineer"


# -----------------------------------------------------------------
# Load Model and Tokenizer
# -----------------------------------------------------------------
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        device_map="cpu",
        dtype=torch.float16
    )

    model.eval()

    return model, tokenizer


# -----------------------------------------------------------------
# Prompt Builder
# -----------------------------------------------------------------
def build_prompt(l2_req, l3_req):

    prompt = f"""
        You are an automotive transmission control software engineer.
        Your task is to derive detailed L4 transmission software requirements from higher-level requirements.
        L4 requirements represent software-level logical rules implemented inside the Transmission Control Unit (TCU).

        ---------------------------------------------------------------------

        RULES

        1. Generate exactly 5 L4 requirements.
        2. Use the syntax: IF <condition> THEN <action>
        3. Number rules from 1 to 5.
        4. Each rule must appear on a new line.
        5. Do not generate explanations.
        6. Do not generate tokens like:
        ### END
        ### MODE
        ### CODE
        ### L4

        ---------------------------------------------------------------------

        EXAMPLE

        L2 Requirement:
        The transmission shall prevent gear shift when the brake pedal is pressed.

        L3 Requirement:
        The TCU shall inhibit gear shift when Brake_Pedal = Pressed.

        L4 Requirements:

        1. IF Brake_Pedal = Pressed THEN Shift_Request = Rejected
        2. IF Shift_Request = Rejected THEN Maintain_Current_Gear
        3. IF Brake_Pedal = Released THEN Enable_Shift_Request
        4. IF Brake_Pedal = Pressed THEN Log_Shift_Inhibition_Event
        5. IF Brake_Pedal = Pressed THEN Disable_Shift_Actuator

        ---------------------------------------------------------------------

        TASK

        L2 Requirement:
        {l2_req}

        L3 Requirement:
        {l3_req}

        L4 Requirements:

        1.
        """

    return prompt


# -----------------------------------------------------------------
# Inference Function
# -----------------------------------------------------------------
def generate_requirements(model, tokenizer, l2_req, l3_req):

    prompt = build_prompt(l2_req, l3_req)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove the prompt from the output
    result = generated_text.replace(prompt, "").strip()

    return result


# -----------------------------------------------------------------
# Local Test
# -----------------------------------------------------------------
if __name__ == "__main__":

    model, tokenizer = load_model()

    l2 = "The transmission shall prevent reverse gear engagement at high speed."
    l3 = "The TCU shall reject reverse gear request when vehicle speed exceeds limit."

    output = generate_requirements(model, tokenizer, l2, l3)

    print("\nGenerated L4 Requirements:\n")
    print(output)
