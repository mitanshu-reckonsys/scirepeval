from transformers import AutoTokenizer
import torch

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
from typing import List, Optional, Dict, Union
from abc import ABC, abstractmethod
import importlib.metadata
import warnings
import json
import logging

logger = logging.getLogger(__name__)
from copy import deepcopy

# Lazy import for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from gritlm import GritLM
    GRITLM_AVAILABLE = True
except ImportError:
    GRITLM_AVAILABLE = False

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False

# Version requirements
MIN_TRANSFORMERS_VERSION_QWEN3 = (4, 51, 0)
MIN_TRANSFORMERS_VERSION_GEMMA = (4, 56, 0)
MIN_SENTENCE_TRANSFORMERS_VERSION = (2, 7, 0)

# Task type constants
SEARCH_TASK_ID = '[SRCH]'
QUERY_TYPE = 'q'
CANDIDATE_TYPE = 'c'

# Field name constants
TITLE_FIELD = 'title'
CONTENT_FIELD = 'content'

# Special token constants
BERT_STYLE_SEP_TOKEN = '[SEP]'


def _parse_version(version_str: str) -> tuple:
    """Parse version string into tuple of integers for comparison."""
    try:
        version_str = version_str.split('-')[0]
        return tuple(int(x) for x in version_str.split('.'))
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _get_package_version(package_name: str) -> str:
    """Get installed version of a package."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _check_version_compatibility(model_type: str) -> tuple:
    """
    Check if the installed transformers version supports the requested model type.

    Returns:
        (is_compatible: bool, error_message: str)
    """
    transformers_version = _get_package_version("transformers")
    current_version = _parse_version(transformers_version)

    if model_type == "qwen3":
        if current_version < MIN_TRANSFORMERS_VERSION_QWEN3:
            return False, (
                f"Qwen3 requires transformers >= {MIN_TRANSFORMERS_VERSION_QWEN3}, "
                f"but you have {transformers_version}. "
                f"Please upgrade: pip install 'transformers>={MIN_TRANSFORMERS_VERSION_QWEN3}'"
            )

    elif model_type == "gemma":
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            return False, (
                f"Gemma requires sentence-transformers to be installed. "
                f"Please install: pip install 'sentence-transformers>={MIN_SENTENCE_TRANSFORMERS_VERSION}'"
            )

        if current_version < MIN_TRANSFORMERS_VERSION_GEMMA:
            warnings.warn(
                f"Gemma works best with transformers >= {MIN_TRANSFORMERS_VERSION_GEMMA}, "
                f"but you have {transformers_version}. Some features may not work correctly.",
                UserWarning
            )

        st_version = _get_package_version("sentence-transformers")
        st_current = _parse_version(st_version)

        if st_current < MIN_SENTENCE_TRANSFORMERS_VERSION:
            warnings.warn(
                f"sentence-transformers >= {MIN_SENTENCE_TRANSFORMERS_VERSION} is recommended, "
                f"but you have {st_version}. Consider upgrading: "
                f"pip install 'sentence-transformers>={MIN_SENTENCE_TRANSFORMERS_VERSION}'",
                UserWarning
            )

    return True, ""


def _merge_prompts(base: Dict, override: Dict) -> Dict:
    """
    Recursively merge two prompt dictionaries.
    Override values take precedence over base values.
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key == "base_prompt":
            # Skip the base_prompt reference in the merged result
            continue
        elif key == "parameters":
            # Merge parameters separately
            if "parameters" in result:
                result["parameters"].update(value)
            else:
                result["parameters"] = deepcopy(value)
        elif isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursively merge nested dictionaries (like [SRCH])
            result[key] = _merge_prompts(result[key], value)
        else:
            # Override the value
            result[key] = deepcopy(value)

    return result


def load_prompts(prompts_data: Dict, prompt_name: str, _visited: Optional[set] = None) -> Dict:
    """
    Load and resolve prompts from the prompts configuration.

    Args:
        prompts_data: The full prompts dictionary loaded from JSON
        prompt_name: The name of the prompt configuration to load
        _visited: Internal parameter to track visited prompts and prevent cycles

    Returns:
        A fully resolved prompt dictionary with all base_prompt references resolved

    Raises:
        ValueError: If prompt_name doesn't exist or if circular reference is detected
    """
    if prompt_name not in prompts_data:
        raise ValueError(f"Prompt configuration '{prompt_name}' not found in prompts data")

    # Track visited prompts to detect circular references
    if _visited is None:
        _visited = set()

    if prompt_name in _visited:
        raise ValueError(f"Circular reference detected: {prompt_name} has already been visited")

    _visited.add(prompt_name)

    prompt_config = prompts_data[prompt_name]

    # If this config has a base_prompt, recursively resolve it first
    if "base_prompt" in prompt_config:
        base_prompt_name = prompt_config["base_prompt"]
        base_config = load_prompts(prompts_data, base_prompt_name, _visited.copy())
        # Merge the base config with the current config
        return _merge_prompts(base_config, prompt_config)
    else:
        # No base prompt, return a deep copy of the config
        return deepcopy(prompt_config)


def load_prompts_from_file(file_path: str, prompt_name: str) -> Dict:
    """
    Load and resolve prompts from a JSON file.

    Args:
        file_path: Path to the JSON file containing prompt configurations
        prompt_name: The name of the prompt configuration to load

    Returns:
        A fully resolved prompt dictionary
    """
    with open(file_path, 'r') as f:
        prompts_data = json.load(f)

    return load_prompts(prompts_data, prompt_name)



class InstructorEmbeddingModel(ABC):

    def __init__(self, embed_model: str, model_type: str, task_prompts: Dict[str, str]):
        is_compatible, error_msg = _check_version_compatibility(model_type)
        if not is_compatible:
            raise ValueError(error_msg)

        self.embed_model = embed_model
        self.task_prompts = task_prompts
        self._task_id = None
        self.task_name = None

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value
        if value is not None:
            self._log_prompts(value)

    def _log_prompts(self, task_id):
        pass

    def _setup_tokenizer_sep_token(self, tokenizer):
        if hasattr(tokenizer, 'eos_token'):
            tokenizer.sep_token = tokenizer.eos_token

    @abstractmethod
    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        pass


class SentenceTransformerModel(InstructorEmbeddingModel):
    """
    Unified wrapper for SentenceTransformer-compatible embedding models.

    Passes prompts via the `prompt=` kwarg on encoder.encode(), which is the
    recommended approach for all ST-compatible models (Gemma, Qwen3, F2LLM,
    Harrier, Jina, etc.). This matches how these models are trained.

    For search tasks, only queries receive a prompt; documents are encoded with
    no prompt. For non-search tasks, all items receive the task prompt.

    task_prompts format (after load_prompts_from_file resolves inheritance):
        {
            "[CLF]": "task: classification | query: ",
            "[RGN]": "task: clustering | query: ",
            "[PRX]": "task: sentence similarity | query: ",
            "[SRCH]": {"q": "task: search result | query: ", "c": ""},
        }
    Prompt strings are plain prefixes — ST appends the text automatically.

    Alternatively, use prompt_name_map to use named presets baked into the model's
    ST config (recommended for models like Gemma, Harrier that define them):
        {
            "[CLF]": "Classification",
            "[PRX]": "STS",
            "[SRCH]": {"q": "Retrieval-query", "c": "Retrieval-document"},
        }
    task_prompts and prompt_name_map are mutually exclusive.

    Args:
        embed_model: HuggingFace model name or path.
        task_prompts: Task prompt prefix strings keyed by task ID.
        prompt_name_map: Task ID -> preset name(s) from the model's ST config.
        ckpt_path: Optional path to a PyTorch checkpoint file or DeepSpeed ZeRO directory.
        truncate_dim: Truncate embeddings to this dimension (passed to SentenceTransformer).
        trust_remote_code: Pass trust_remote_code=True to SentenceTransformer.
        max_seq_length: Override the encoder's max_seq_length after loading.
    """

    def __init__(
        self,
        embed_model: str,
        task_prompts: Dict[str, str] = None,
        prompt_name_map: Dict[str, Union[str, dict]] = None,
        ckpt_path: str = None,
        truncate_dim: int = None,
        trust_remote_code: bool = False,
        max_seq_length: int = None,
    ):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformerModel requires sentence-transformers. "
                "Please install: pip install sentence-transformers"
            )
        if task_prompts and prompt_name_map:
            raise ValueError("Specify either task_prompts or prompt_name_map, not both.")
        super().__init__(embed_model, "sentence_transformer", task_prompts or {})

        self.device = _get_device()
        self.prompt_name_map = prompt_name_map or {}

        st_kwargs = {}
        if truncate_dim is not None:
            st_kwargs["truncate_dim"] = truncate_dim
        if trust_remote_code:
            st_kwargs["trust_remote_code"] = True

        self.encoder = SentenceTransformer(embed_model, **st_kwargs)

        if max_seq_length is not None:
            self.encoder.max_seq_length = max_seq_length

        if ckpt_path:
            import os
            if os.path.isdir(ckpt_path):
                from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
                state_dict = get_fp32_state_dict_from_zero_checkpoint(ckpt_path)
                encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
                self.encoder[0].auto_model.load_state_dict(encoder_state)
            else:
                checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                encoder_state = {k.replace('encoder.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('encoder.')}
                self.encoder[0].auto_model.load_state_dict(encoder_state)

        self.tokenizer = self.encoder.tokenizer
        self._setup_tokenizer_sep_token(self.tokenizer)

    def _log_prompts(self, task_id):
        key = self.task_name or (SEARCH_TASK_ID if isinstance(task_id, dict) else task_id)
        if isinstance(task_id, dict):
            prompts = {t: self._get_encode_kwargs(t) for t in (QUERY_TYPE, CANDIDATE_TYPE)}
        else:
            prompts = self._get_encode_kwargs()
        logger.info(f"Prompts for task {key!r}: {prompts}")

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode(formatted_batch, convert_to_tensor=True, device=self.device)

    def _get_encode_kwargs(self, batch_type: str = None) -> dict:
        """Return prompt kwargs for encoder.encode() for the current task and batch_type."""
        # For search tasks (task_id is a dict), use task_name if set, else fall back to [SRCH]
        key = self.task_name or (SEARCH_TASK_ID if isinstance(self.task_id, dict) else self.task_id)

        if self.prompt_name_map and key is not None:
            entry = self.prompt_name_map.get(key)
            if entry is not None:
                name = entry.get(batch_type or QUERY_TYPE) if isinstance(entry, dict) else entry
                if name:
                    return {"prompt_name": name}
            return {}

        if self.task_prompts and key is not None:
            entry = self.task_prompts.get(key)
            if entry is not None:
                p = entry.get(batch_type or QUERY_TYPE, "") if isinstance(entry, dict) else entry
                if p:
                    return {"prompt": p}
        return {}

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        is_search_task = isinstance(self.task_id, dict)

        if not is_search_task:
            return self.encoder.encode(batch, convert_to_tensor=True, device=self.device,
                                       **self._get_encode_kwargs())

        # Search task: queries get a prompt, documents get none
        query_texts, query_indices, doc_texts, doc_indices = [], [], [], []
        for i, (_, batch_type) in enumerate(batch_ids):
            if batch_type == QUERY_TYPE:
                query_texts.append(batch[i])
                query_indices.append(i)
            else:
                doc_texts.append(batch[i])
                doc_indices.append(i)

        embeddings = [None] * len(batch)
        if query_texts:
            for idx, emb in zip(query_indices, self.encoder.encode(query_texts, convert_to_tensor=True, device=self.device,
                                                                    **self._get_encode_kwargs(QUERY_TYPE))):
                embeddings[idx] = emb
        if doc_texts:
            for idx, emb in zip(doc_indices, self.encoder.encode(doc_texts, convert_to_tensor=True, device=self.device,
                                                                   **self._get_encode_kwargs(CANDIDATE_TYPE))):
                embeddings[idx] = emb
        return torch.stack(embeddings)



VOYAGE4_NANO_MODEL = "voyageai/voyage-4-nano"


class Voyage4Model(InstructorEmbeddingModel):
    """
    Voyage4 model wrapper supporting both local (SentenceTransformer) and API modes.

    Local mode delegates to SentenceTransformerModel.
    API mode uses the Voyage API for document encoding and the local nano model for queries.

    Args:
        embed_model: HuggingFace model path (local) or Voyage model name (API).
        task_prompts: Ignored - Voyage4 uses its own internal prompts.
        ckpt_path: Ignored.
        use_api: If True, use the Voyage API for docs and local nano model for queries.
        truncate_dim: Embedding truncation dimension.
    """

    def __init__(
        self,
        embed_model: str,
        task_prompts: Dict[str, str] = None,
        ckpt_path: str = None,
        use_api: bool = False,
        truncate_dim: int = 1024,
    ):
        super().__init__(embed_model, "voyage4", task_prompts or {})

        self.use_api = use_api
        self.device = _get_device()

        if use_api:
            if not VOYAGEAI_AVAILABLE:
                raise ImportError(
                    "Voyage4 API mode requires the voyageai package. "
                    "Please install: pip install voyageai\n"
                    "You also need to set the VOYAGE_API_KEY environment variable."
                )
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "Voyage4 API mode requires sentence-transformers for local query encoding. "
                    "Please install: pip install sentence-transformers"
                )
            self.client = voyageai.Client()
            self.doc_model = embed_model
            self.query_encoder = SentenceTransformer(VOYAGE4_NANO_MODEL, trust_remote_code=True, truncate_dim=truncate_dim)
            self.query_encoder.max_seq_length = 512
            self.tokenizer = AutoTokenizer.from_pretrained(f"voyageai/{embed_model}")
        else:
            self._local = SentenceTransformerModel(
                embed_model,
                task_prompts,
                truncate_dim=truncate_dim,
                trust_remote_code=True,
                max_seq_length=512,
            )
            self.tokenizer = self._local.tokenizer

    def _embed_api(self, texts: List[str]) -> torch.Tensor:
        result = self.client.embed(texts=texts, model=self.doc_model, input_type="document")
        return torch.tensor(result.embeddings)

    def _encode_queries_local(self, texts: List[str]) -> torch.Tensor:
        return self.query_encoder.encode_query(texts, convert_to_tensor=True, device=self.device)

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        # Only used in API mode for non-search tasks
        return self._embed_api(formatted_batch)

    def _encode_batch_api(self, batch: List[str], batch_ids: Optional[List] = None) -> torch.Tensor:
        is_search_task = isinstance(self.task_id, dict)

        if is_search_task and batch_ids:
            query_texts, query_indices, doc_texts, doc_indices = [], [], [], []
            for i, (_, batch_type) in enumerate(batch_ids):
                if batch_type == QUERY_TYPE:
                    query_texts.append(batch[i])
                    query_indices.append(i)
                else:
                    doc_texts.append(batch[i])
                    doc_indices.append(i)

            embeddings = [None] * len(batch)
            if query_texts:
                for idx, emb in zip(query_indices, self._encode_queries_local(query_texts)):
                    embeddings[idx] = emb
            if doc_texts:
                for idx, emb in zip(doc_indices, self._embed_api(doc_texts)):
                    embeddings[idx] = emb
            return torch.stack(embeddings)
        else:
            return self._embed_api(batch)

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        if not self.use_api:
            self._local.task_id = self.task_id
            self._local.task_name = self.task_name
            return self._local(batch, batch_ids)
        else:
            return self._encode_batch_api(batch, batch_ids)


class GritLMModel(InstructorEmbeddingModel):

    def __init__(self, embed_model: str, task_prompts: Dict[str, str], ckpt_path: str = None, **kwargs):
        super().__init__(embed_model, "gritlm", task_prompts)

        if not GRITLM_AVAILABLE:
            raise ImportError(
                "GritLM requires the gritlm package. "
                "Please install: pip install gritlm"
            )

        self.encoder = GritLM(self.embed_model, torch_dtype="auto", mode="embedding")
        self.tokenizer = self.encoder.tokenizer
        self._setup_tokenizer_sep_token(self.tokenizer)

    @staticmethod
    def _gritlm_instruction(instruction: str) -> str:
        if instruction:
            return f"<|user|>\n{instruction}\n<|embed|>\n"
        else:
            return "<|embed|>\n"

    def _log_prompts(self, task_id):
        is_search_task = isinstance(task_id, dict)
        if is_search_task:
            q = self._gritlm_instruction(self.task_prompts.get(SEARCH_TASK_ID, {}).get(QUERY_TYPE, ''))
            c = self._gritlm_instruction("")
            logger.info(f"Prompts for task {SEARCH_TASK_ID!r}: query={q!r}, candidate={c!r}")
        else:
            instruction = self._gritlm_instruction(self.task_prompts.get(task_id, ""))
            logger.info(f"Prompt for task {task_id!r}: {instruction!r}")

    def _encode_batch(self, formatted_batch: List[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(formatted_batch, convert_to_tensor=True)
        return embeddings

    def __call__(self, batch: List[str], batch_ids: Optional[List] = None):
        is_search_task = isinstance(self.task_id, dict)

        if not is_search_task:
            instruction = self.task_prompts.get(self.task_id, "")
            return self.encoder.encode(batch, instruction=self._gritlm_instruction(instruction), convert_to_tensor=True)
        else:
            query_instruction = self._gritlm_instruction(self.task_prompts[SEARCH_TASK_ID].get(QUERY_TYPE, ''))
            candidate_instruction = self._gritlm_instruction("")

            instructions = [
                query_instruction if batch_type == QUERY_TYPE else candidate_instruction
                for _, batch_type in batch_ids
            ]

            return self.encoder.encode(batch, instruction=instructions, convert_to_tensor=True)
