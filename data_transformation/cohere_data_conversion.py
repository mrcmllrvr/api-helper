import yaml
import json
import re

def get_schema_details(schema_ref, all_schemas):
    """
    Recursively fetches and formats details for a schema reference.
    This helps in detailing the request body or response objects.
    """
    if not schema_ref:
        return "No schema reference provided."
        
    schema_name = schema_ref.split('/')[-1]
    schema = all_schemas.get(schema_name, {})
    
    properties = schema.get('properties', {})
    if not properties:
        schema_type = schema.get('type', 'object')
        return f"A `{schema_type}` object is expected."

    details = []
    for prop_name, prop_data in properties.items():
        prop_type = prop_data.get('type', 'N/A')
        description = prop_data.get('description', 'No description.').strip()
        
        description = re.sub(r'\s+', ' ', description)
        
        required_list = schema.get('required', [])
        req_status = "required" if prop_name in required_list else "optional"

        details.append(f"- **`{prop_name}`** ({prop_type}, {req_status}): {description}")
    
    if not details:
        return "No specific properties defined."
        
    return "\n".join(details)


def process_openapi_spec(file_path):
    """
    Parses a Cohere OpenAPI YAML file and chunks it by API operation,
    creating a structured list of documents ready for embedding.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            spec = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading or parsing YAML file: {e}")
        return []

    documents = []
    all_schemas = spec.get('components', {}).get('schemas', {})
    all_params_components = spec.get('components', {}).get('parameters', {})

    for path, path_item in spec.get('paths', {}).items():
        for method, operation_or_list in path_item.items():
            operations = operation_or_list if isinstance(operation_or_list, list) else [operation_or_list]
            
            for operation in operations:
                if not isinstance(operation, dict):
                    continue
            
            operation_id = operation.get('operationId') or operation.get('summary')
            if not operation_id:
                continue

            content_parts = []
            
            summary = operation.get('summary', operation_id) # Use operationId as fallback for summary
            description = operation.get('description', 'No description provided.')
            tags = ", ".join(operation.get('tags', []))

            content_parts.append(f"# Operation: {summary}")
            content_parts.append(f"**Endpoint:** `{method.upper()} {path}`")
            if tags:
                content_parts.append(f"**Tags:** {tags}")
            if description:
                content_parts.append(f"\n**Description:**\n{description.strip()}")

            # --- Parameters (query, path, header) ---
            params = operation.get('parameters', [])
            if params:
                content_parts.append("\n**Parameters:**")
                for param_ref in params:
                    param = {}
                    # Resolve parameter references if they exist
                    if '$ref' in param_ref:
                        param_name = param_ref['$ref'].split('/')[-1]
                        param = all_params_components.get(param_name, {})
                    else:
                        param = param_ref
                    
                    if not param: continue

                    name = param.get('name')
                    loc = param.get('in')
                    req = "required" if param.get('required') else "optional"
                    param_desc = param.get('description', 'No description.').strip()
                    param_desc = re.sub(r'\s+', ' ', param_desc)
                    schema_type = param.get('schema', {}).get('type', 'string')
                    content_parts.append(f"- **`{name}`** ({loc}, {req}, {schema_type}): {param_desc}")
            
            # --- Request Body ---
            request_body = operation.get('requestBody')
            if request_body:
                content_parts.append("\n**Request Body:**")
                content = request_body.get('content', {})
                for media_type, media_details in content.items():
                    schema_ref = media_details.get('schema', {}).get('$ref')
                    body_details = get_schema_details(schema_ref, all_schemas)
                    content_parts.append(f"The request body is a `{media_type}` object with the following key properties:\n{body_details}")

            # --- Responses ---
            responses = operation.get('responses', {})
            # Look for a 2xx success response code
            success_code = next((code for code in responses if code.startswith('2')), None)
            if success_code:
                response_desc = responses[success_code].get('description', 'Successful operation.')
                content_parts.append(f"\n**Successful Response ({success_code}):**\n{response_desc.strip()}")

            
            examples = operation.get('x-fern-examples', [])
            if examples:
                content_parts.append("\n**Code Examples:**")
                for example in examples:
                    code_samples = example.get('code-samples', [])
                    for sample in code_samples:
                        lang = sample.get('language')
                        code = sample.get('code')
                        if lang and code:
                            lang_map = {'curl': 'sh', 'node': 'javascript', 'typescript': 'typescript'}
                            md_lang = lang_map.get(lang.lower(), lang.lower())
                            content_parts.append(f"\n**{lang.capitalize()}:**\n```{md_lang}\n{code.strip()}\n```")
            
            document_content = "\n\n".join(content_parts)

            metadata = {
                "operationId": operation_id,
                "path": path,
                "method": method.upper(),
                "summary": summary,
                "tags": operation.get('tags', [])
            }

            documents.append({
                "content": document_content,
                "metadata": metadata
            })

    return documents


if __name__ == "__main__":
    input_filepath = 'data/cohere-openapi.yaml'
    output_filepath = 'cohere.json'
    
    processed_documents = process_openapi_spec(input_filepath)
    
    if processed_documents:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_documents, f, indent=2)
        
        print(f"Successfully processed {len(processed_documents)} API operations.")
        print(f"Output saved to '{output_filepath}'")
        print("\n--- Sample of the first processed document ---")
        print(json.dumps(processed_documents[0], indent=2))
        print("---------------------------------------------")