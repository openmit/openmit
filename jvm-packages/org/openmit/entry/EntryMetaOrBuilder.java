// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: entry_meta.proto

package org.openmit.entry;

public interface EntryMetaOrBuilder extends
    // @@protoc_insertion_point(interface_extends:mit.protobuf.EntryMeta)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */
  int getEntryMetaMapCount();
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */
  boolean containsEntryMetaMap(
      java.lang.String key);
  /**
   * Use {@link #getEntryMetaMapMap()} instead.
   */
  @java.lang.Deprecated
  java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray>
  getEntryMetaMap();
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */
  java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray>
  getEntryMetaMapMap();
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */

  org.openmit.entry.FieldIdArray getEntryMetaMapOrDefault(
      java.lang.String key,
      org.openmit.entry.FieldIdArray defaultValue);
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */

  org.openmit.entry.FieldIdArray getEntryMetaMapOrThrow(
      java.lang.String key);

  /**
   * <pre>
   * embedding size 
   * </pre>
   *
   * <code>uint32 embedding_size = 2;</code>
   */
  int getEmbeddingSize();

  /**
   * <pre>
   * model name 
   * </pre>
   *
   * <code>string model = 3;</code>
   */
  java.lang.String getModel();
  /**
   * <pre>
   * model name 
   * </pre>
   *
   * <code>string model = 3;</code>
   */
  com.google.protobuf.ByteString
      getModelBytes();
}