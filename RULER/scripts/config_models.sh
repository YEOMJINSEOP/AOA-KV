# Model registry for RULER runs. One file for every sequence length: the
# sequence length is a run-script argument, not a property of the model.
#
# MODEL_SELECT <model_name> echoes
#   MODEL_PATH:MODEL_TEMPLATE_TYPE:MODEL_FRAMEWORK:TOKENIZER_PATH:TOKENIZER_TYPE
#
# Template types must stay exactly as upstream RULER had them — they decide the
# prompt wrapper, so changing one makes numbers incomparable with earlier runs.

TEMPERATURE="0.0"   # greedy
TOP_P="1.0"
TOP_K="32"

MODEL_SELECT() {
    MODEL_NAME=$1

    case $MODEL_NAME in
        llama-3.1-8b-instruct)
            MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            ;;
        llama-3.2-3b-instruct)
            MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            ;;
        llama-3.2-1b-instruct)
            MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
            MODEL_TEMPLATE_TYPE="meta-llama3"
            ;;
        mistral-7b-instruct-v0.2)
            MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2"
            MODEL_TEMPLATE_TYPE="meta-chat"
            ;;
        longchat-7b-v1.5-32k)
            MODEL_PATH="lmsys/longchat-7b-v1.5-32k"
            MODEL_TEMPLATE_TYPE="lwm-chat"
            ;;
        qwen2.5-7b-instruct)
            MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
            MODEL_TEMPLATE_TYPE="qwen-chat"
            ;;
        qwen2.5-3b-instruct)
            MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
            MODEL_TEMPLATE_TYPE="qwen-chat"
            ;;
        *)
            echo ""
            return
            ;;
    esac

    # Everything here runs through HF transformers with the model's own tokenizer.
    echo "$MODEL_PATH:$MODEL_TEMPLATE_TYPE:hf:$MODEL_PATH:hf"
}
