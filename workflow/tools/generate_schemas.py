from pathlib import Path
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bertopic_workflow.common import write_json  # noqa: E402
from bertopic_workflow.config import WorkflowConfig  # noqa: E402

events = ['capabilities', 'schema', 'schema_written', 'doctor', 'inspected', 'config_validated',
          'job_status', 'cancel_requested', 'release_verified', 'package_created', 'run_started',
          'stage_started', 'stage_progress', 'stage_completed', 'run_prepared', 'validation_started',
          'run_completed', 'run_failed', 'run_cancelled', 'error']
directory = ROOT / 'src/bertopic_workflow/schemas'
project = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['project']
write_json(directory / 'runtime_requirements.json', {'version': project['version'], 'dependencies': project['dependencies'], 'extras': project.get('optional-dependencies', {})})
config_schema = WorkflowConfig.model_json_schema()
config_schema['$schema'] = 'https://json-schema.org/draft/2020-12/schema'
write_json(directory / 'workflow.schema.json', config_schema)
write_json(directory / 'event.schema.json', {
    '$schema': 'https://json-schema.org/draft/2020-12/schema', 'title': 'BERTopic workflow JSONL v1',
    'type': 'object', 'additionalProperties': False,
    'required': ['schema_version', 'event', 'run_id', 'sequence', 'at', 'data'],
    'properties': {'schema_version': {'const': 1}, 'event': {'enum': events},
                   'run_id': {'type': ['string', 'null']}, 'sequence': {'type': 'integer', 'minimum': 1},
                   'at': {'type': 'string', 'format': 'date-time'}, 'data': {'type': 'object'}}})
print('Generated configuration and event schemas.')
