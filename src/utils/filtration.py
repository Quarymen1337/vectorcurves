def filter_json_data(data_list):
    valid_data = []
    removed_count = 0
    
    for item in data_list:
        try:
            coords = get_ca_coordinates(item)
            if coords is not None and len(coords) >= 2:
                valid_data.append(item)
            else:
                removed_count += 1
                
        except Exception:
            removed_count += 1
            continue
            
    print(f"Обработано: {len(data_list)}. Удалено: {removed_count}. Осталось: {len(valid_data)}")
    return valid_data