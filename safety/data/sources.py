"""Convert downloaded annotation tables with explicit field and label mappings."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import yaml

from safety.data.schema import normalize_record, read_jsonl, write_jsonl


def nested_value(row, key):
    current = row
    for part in key.split('.'):
        if isinstance(current, list):
            if not part.isdigit() or int(part) >= len(current):
                raise KeyError(key)
            current = current[int(part)]
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(key)
    return current


def stream_source(path, records_key=None, delimiter=','):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in ('.jsonl', '.ndjson'):
        yield from read_jsonl(path)
    elif suffix in ('.csv', '.tsv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t' if suffix == '.tsv' else delimiter)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError(f'{path} needs unique column names.')
            for number, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f'{path}:{number}: too many CSV columns.')
                yield number, row
    elif suffix == '.json':
        value = json.loads(path.read_text(encoding='utf-8'))
        values = nested_value(value, records_key) if records_key else value
        if not isinstance(values, list):
            raise ValueError('JSON input must be a list or use records_key to locate one.')
        for number, row in enumerate(values, 1):
            if not isinstance(row, dict):
                raise ValueError(f'{path}:{number}: expected an annotation object.')
            yield number, row
    else:
        raise ValueError(f'Unsupported annotation suffix {suffix}.')


def load_mapping(path):
    mapping = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(mapping, dict) or not isinstance(mapping.get('fields'), dict):
        raise ValueError('The mapping needs a fields dictionary.')
    if 'prompt' not in mapping['fields']:
        raise ValueError('A source prompt mapping is required.')
    if not mapping.get('source'):
        raise ValueError('Name the source dataset in the mapping.')
    for destination, origin in mapping['fields'].items():
        if not isinstance(origin, str) or not origin:
            raise ValueError(f'Invalid source field for {destination}.')
    for field in ('decision', 'level', 'cf_decision'):
        table = mapping.get('values', {}).get(field, {})
        if not isinstance(table, dict):
            raise ValueError(f'values.{field} must be a label mapping.')
    return mapping


def map_value(value, field, mapping):
    values = mapping.get('values', {}).get(field)
    if values is not None:
        key = str(value)
        table = {str(name): result for name, result in values.items()}
        if key not in table:
            raise ValueError(f'Unmapped {field} label {key!r}.')
        value = table[key]
    if field in ('decision', 'cf_decision', 'level'):
        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, str) and value.strip().isdigit():
            value = int(value)
    return value


def convert_row(row, mapping, origin, number):
    record = dict(mapping.get('defaults', {}))
    optional = set(mapping.get('optional', []))
    for field, source in mapping['fields'].items():
        try:
            value = nested_value(row, source)
        except KeyError:
            if field in optional:
                continue
            raise ValueError(f'Missing source field {source}.') from None
        if value is None or value == '':
            if field in optional:
                continue
            raise ValueError(f'Empty source field {source}.')
        record[field] = map_value(value, field, mapping)
    record['source'] = mapping['source']
    record.setdefault('language', mapping.get('language', 'ko'))
    record = normalize_record(record, mapping.get('stage', 'seed'))
    record['provenance'] = {'file': str(origin), 'row': number, 'source': mapping['source']}
    return record


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def convert_sources(paths, mapping, allow_rejections=False):
    converted, rejected = [], []
    identities = set()
    source_counts = Counter()
    for path in paths:
        for number, row in stream_source(path, mapping.get('records_key'), mapping.get('delimiter', ',')):
            try:
                record = convert_row(row, mapping, path, number)
                if record['id'] in identities:
                    raise ValueError(f'Duplicate converted id {record["id"]}.')
            except (ValueError, TypeError, KeyError) as error:
                rejected.append({'file': str(path), 'row': number, 'error': str(error)})
                continue
            identities.add(record['id'])
            converted.append(record)
            source_counts[str(path)] += 1
    report = {
        'source': mapping['source'],
        'stage': mapping.get('stage', 'seed'),
        'input_sha256': {str(path): file_digest(path) for path in paths},
        'accepted': len(converted),
        'rejected': rejected,
        'per_file': dict(source_counts),
        'levels': dict(Counter(str(row['level']) for row in converted)),
    }
    report['publishable'] = bool(converted) and (allow_rejections or not rejected)
    return converted, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--allow-rejections', action='store_true')
    args = parser.parse_args(argv)
    inputs = [Path(path).resolve() for path in args.input]
    output, report_path = Path(args.output).resolve(), Path(args.report).resolve()
    if output in inputs or report_path in inputs or output == report_path:
        parser.error('Inputs, converted records and the report must have separate paths.')
    mapping = load_mapping(args.mapping)
    rows, report = convert_sources(inputs, mapping, args.allow_rejections)
    report['mapping_sha256'] = file_digest(args.mapping)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if not report['publishable']:
        raise ValueError(f'Conversion requires review. See {report_path}.')
    write_jsonl(output, rows)
    print(json.dumps({'records': len(rows), 'output': str(output), 'report': str(report_path)}))
