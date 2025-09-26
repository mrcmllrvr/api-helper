import yaml
import json
import re

def get_schema_details(schema_ref, all_schemas):
    """
    Recursively fetches and formats details for a schema reference.
    This helps in detailing the request body or response objects.
    """
    # Extract the schema name from the reference path
    schema_name = schema_ref.split('/')[-1]
    schema = all_schemas.get(schema_name, {})
    
    properties = schema.get('properties', {})
    if not properties:
        return "No specific properties defined."

    details = []
    for prop_name, prop_data in properties.items():
        prop_type = prop_data.get('type', 'N/A')
        description = prop_data.get('description', 'No description.').strip()
        
        description = re.sub(r'\s+', ' ', description)
        details.append(f"- `{prop_name}` ({prop_type}): {description}")
    
    return "\n".join(details)


def process_openapi_spec(file_path):
    """
    Parses an OpenAPI YAML file and chunks it by API operation,
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

    # Iterate over each path
    for path, path_item in spec.get('paths', {}).items():
        # Iterate over each HTTP method (get, post, delete) for the path
        for method, operation in path_item.items():
            if 'operationId' not in operation:
                continue

            
            content_parts = []
            
            operation_id = operation.get('operationId')
            summary = operation.get('summary', 'No summary provided.')
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
                for param in params:
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
                    if schema_ref:
                        body_details = get_schema_details(schema_ref, all_schemas)
                        content_parts.append(f"The request body is a `{media_type}` object with the following key properties:\n{body_details}")
                    else:
                        content_parts.append(f"A `{media_type}` object is required.")

            # --- Responses ---
            responses = operation.get('responses', {})
            if '200' in responses:
                response_desc = responses['200'].get('description', 'Successful operation.')
                content_parts.append(f"\n**Successful Response (200 OK):**\n{response_desc.strip()}")

            # --- Code Examples ---
            examples = operation.get('x-oaiMeta', {}).get('examples', [])
            if examples:
                content_parts.append("\n**Code Examples:**")
                for example in examples:
                    # Handle both simple and language-specific examples
                    if isinstance(example, dict) and 'request' in example:
                        req = example['request']
                        if isinstance(req, dict):
                            for lang, code in req.items():
                                if lang in ['curl', 'python', 'node.js', 'javascript', 'go', 'java', 'ruby', 'csharp']:
                                    lang_key = lang.lower().replace('node.js', 'javascript')
                                    content_parts.append(f"\n**{lang.capitalize()}:**\n```{lang_key}\n{code.strip()}\n```")
                    elif isinstance(example, str):
                        content_parts.append(f"\n```\n{example.strip()}\n```") 
                    
                    # if 'request' in example and isinstance(example['request'], dict):
                    #     for lang, code in example['request'].items():
                    #          if lang in ['curl', 'python', 'node.js', 'javascript', 'go', 'java', 'ruby', 'csharp']:
                    #             content_parts.append(f"\n**{lang.capitalize()}:**\n```{lang.lower().replace('node.js','javascript')}\n{code.strip()}\n```")
                    
            
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
    input_filepath = 'data/openai-api.yml'
    output_filepath = 'openai.json'
    
    processed_documents = process_openapi_spec(input_filepath)
    
    if processed_documents:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(processed_documents, f, indent=2)
        
        print(f"Successfully processed {len(processed_documents)} API operations.")
        print(f"Output saved to '{output_filepath}'")
        print("\n--- Sample of the first processed document ---")
        print(json.dumps(processed_documents[0], indent=2))
        print("---------------------------------------------")