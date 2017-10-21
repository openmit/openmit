#include "openmit/entity/entry_meta.h"

namespace mit {

EntryMeta::EntryMeta(const mit::ModelParam & model_param) {
  model = model_param.model;
  if (model == "fm" || model == "ffm") {
    embedding_size = model_param.embedding_size;
  }
  LOG(INFO) << "model_param.model: " << model_param.model;
  LOG(INFO) << "model_param.data_format: " << model_param.data_format;
  LOG(INFO) << "model_param.field_combine_set: " << model_param.field_combine_set;
  LOG(INFO) << "model_param.field_combine_pair: " << model_param.field_combine_pair;

  if (model_param.data_format == "libfm" && model == "ffm") {
    if (model_param.field_combine_set == "" && model_param.field_combine_pair == "") {
      LOG(FATAL) << "parameter field_combine_XXX both empty.";
    }
    if (model_param.field_combine_set != "" && model_param.field_combine_pair != "") {
      LOG(FATAL) << "parameter field_combine_XXX both value.";
    }
    if (model_param.field_combine_set != "") {
      ProcessFieldCombineSet(model_param.field_combine_set);
    } else if (model_param.field_combine_pair != "") {
      ProcessFieldCombinePair(model_param.field_combine_pair);
    } else {
      LOG(FATAL) << "parameter field_combine_XXX both empty.";
    }
  }
}

EntryMeta::~EntryMeta() {
  for (auto & kv : fields_map) {
    if (kv.second) { 
      kv.second->clear();
      delete kv.second;
      kv.second = NULL; 
    }
  }
}

int EntryMeta::FieldIndex(const mit_uint & fieldid, const mit_uint & rfieldid) {
  if (fields_map.find(fieldid) == fields_map.end()) {
    return -1;
  } else {
    auto * related_fields_list = fields_map[fieldid];
    for (auto i = 0u; i < related_fields_list->size(); ++i) {
      if ((*related_fields_list)[i] == rfieldid) return i;
    }
    return -1;
  }
}
std::vector<mit_uint> * EntryMeta::CombineInfo(const mit_uint & fieldid) {
  if (fields_map.find(fieldid) == fields_map.end()) {
    return new std::vector<mit_uint>;
  } else {
    return fields_map[fieldid];
  }
}

void EntryMeta::ProcessFieldCombineSet(const std::string field_combine_set) {
  std::vector<std::string> field_items;
  mit::string::Split(field_combine_set, &field_items, ',');
  std::vector<mit_uint> fields(field_items.size(), 0l);
  for (auto i = 0u; i < field_items.size(); ++i) {
    fields[i] = mit::StringToNum<mit_uint>(field_items[i]);
  }
  sort(fields.begin(), fields.end());
  for (auto i = 0u; i < fields.size(); ++i) {
    fields_map.insert(std::make_pair(
      fields.at(i), 
      new std::vector<mit_uint>(fields.begin(), fields.end())));
  }
} 

void EntryMeta::ProcessFieldCombinePair(const std::string field_combine_pair) {
  std::vector<std::string> field_pairs;
  mit::string::Split(field_combine_pair, &field_pairs, ',');
  for (auto i = 0u; i < field_pairs.size(); ++i) {
    std::vector<std::string> items;
    mit::string::Split(field_pairs[i], &items, '^');
    if (items.size() != 2) { 
      LOG(FATAL) << "field_combine_pair format error: " << field_pairs[i];
      continue;
    }
    auto field1 = mit::StringToNum<mit_uint>(items[0]);
    auto field2 = mit::StringToNum<mit_uint>(items[1]);
    FillFieldInfo(field1, field2);
    FillFieldInfo(field2, field1);
  }
  // sort 
  for (auto & kv : fields_map) {
    sort(kv.second->begin(), kv.second->end());
  }
}

void EntryMeta::FillFieldInfo(mit_uint & field1, mit_uint & field2) {
  if (fields_map.find(field1) == fields_map.end()) {
    std::vector<mit_uint> * fields = new std::vector<mit_uint>;
    fields->push_back(field1);
    fields->push_back(field2);
    fields_map.insert(std::make_pair(field1, fields));
  } else {
    fields_map[field1]->push_back(field2);
  }
}

}// namespace mit
