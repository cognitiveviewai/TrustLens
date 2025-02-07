def send_payload(tenant_id: str, client_id: str, control_id: str, owner_id: str, collected_by: str, authorized_user: str, shared_with: list, sensitivity: str, name: str, description: str, type: str, version: str, source: str, attachment: str, data: dict):
    payload = {

        "tenant_id": tenant_id,
        "client_id": client_id,
        "control_id": control_id,
        "owner_id": owner_id,
        "collected_by": collected_by,
        "authorized_user": authorized_user,
        "shared_with": shared_with,
        "sensitivity": sensitivity,
        "name": name,
        "description": description,
        "type": type,
        "version": version,
        "source": source,
        "attachment": attachment,
        "data": data
}
    return payload
