#include "openmit/entry/entry_meta.h"

namespace mit {
EntryMeta::EntryMeta(const mit::ModelParam& model_param) {
  model = model_param.model;
  if (model == "fm" || model == "ffm") {
    embedding_size = model_param.embedding_size;
  }

  if (model == "ffm") {
    if (model_param.field_combine_set == "" 
        && model_param.field_combine_pair == "") {
      LOG(FATAL) << "parameter field_combine_XXX both empty.";
    }
    if (model_param.field_combine_set != "" 
        && model_param.field_combine_pair != "") {
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

void EntryMeta::Save(dmlc::Stream * fo) {
  // model 
  size_t bytes_model = model.size();
  fo->Write((char *) &bytes_model, sizeof(size_t));
  fo->Write(model.c_str(), bytes_model);
  fo->Write((char *) &embedding_size, sizeof(size_t));
  // write fields_map 
  size_t fields_map_size = fields_map.size();
  fo->Write((char *) &fields_map_size, sizeof(size_t));
  mit::entrymeta_map_type::iterator iter = fields_map.begin();
  while (iter != fields_map.end()) {
    fo->Write((char *) &iter->first, sizeof(mit_uint));
    size_t vec_size = iter->second->size();
    fo->Write((char *) &vec_size, sizeof(size_t));
    for (auto i = 0u; i < vec_size; ++i) {
      fo->Write((char *) &iter->second->at(i), sizeof(mit_uint));
    }
    iter++;
  }
} // method Save

void EntryMeta::Load(dmlc::Stream * fi) {
  size_t numerical;
  // read : size of model string
  fi->Read(&numerical, sizeof(size_t));
  char * modelbytes = new char[numerical];
  // read : model string
  fi->Read(modelbytes, numerical);
  model.assign(modelbytes, numerical);
  // read : embedding_size 
  fi->Read(&numerical, sizeof(size_t));
  embedding_size = numerical;
  // read fields map
  fi->Read(&numerical, sizeof(size_t));
  for (size_t i = 0; i < numerical; ++i) {
    mit_uint fid; 
    fi->Read(&fid, sizeof(mit_uint));
    size_t vecsize;
    fi->Read(&vecsize, sizeof(size_t));
    std::vector<mit_uint> * vec = new std::vector<mit_uint>(vecsize);
    mit_uint rfield_elem;
    for (size_t j = 0; j < vecsize; ++j) {
      fi->Read(&rfield_elem, sizeof(mit_uint));
      (*vec)[j] = rfield_elem;
    }
    fields_map.insert(std::make_pair(fid, vec));
  }
  LOG(INFO) << "load entry meta done. model: " << model 
    << ", fields_map.size: " << numerical;
}

}// namespace mit
