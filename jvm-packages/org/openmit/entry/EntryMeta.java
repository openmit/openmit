// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: entry_meta.proto

package org.openmit.entry;

/**
 * Protobuf type {@code mit.protobuf.EntryMeta}
 */
public  final class EntryMeta extends
    com.google.protobuf.GeneratedMessageV3 implements
    // @@protoc_insertion_point(message_implements:mit.protobuf.EntryMeta)
    EntryMetaOrBuilder {
private static final long serialVersionUID = 0L;
  // Use EntryMeta.newBuilder() to construct.
  private EntryMeta(com.google.protobuf.GeneratedMessageV3.Builder<?> builder) {
    super(builder);
  }
  private EntryMeta() {
    embeddingSize_ = 0;
    model_ = "";
  }

  @java.lang.Override
  public final com.google.protobuf.UnknownFieldSet
  getUnknownFields() {
    return this.unknownFields;
  }
  private EntryMeta(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    this();
    if (extensionRegistry == null) {
      throw new java.lang.NullPointerException();
    }
    int mutable_bitField0_ = 0;
    com.google.protobuf.UnknownFieldSet.Builder unknownFields =
        com.google.protobuf.UnknownFieldSet.newBuilder();
    try {
      boolean done = false;
      while (!done) {
        int tag = input.readTag();
        switch (tag) {
          case 0:
            done = true;
            break;
          default: {
            if (!parseUnknownFieldProto3(
                input, unknownFields, extensionRegistry, tag)) {
              done = true;
            }
            break;
          }
          case 10: {
            if (!((mutable_bitField0_ & 0x00000001) == 0x00000001)) {
              entryMetaMap_ = com.google.protobuf.MapField.newMapField(
                  EntryMetaMapDefaultEntryHolder.defaultEntry);
              mutable_bitField0_ |= 0x00000001;
            }
            com.google.protobuf.MapEntry<java.lang.String, org.openmit.entry.FieldIdArray>
            entryMetaMap__ = input.readMessage(
                EntryMetaMapDefaultEntryHolder.defaultEntry.getParserForType(), extensionRegistry);
            entryMetaMap_.getMutableMap().put(
                entryMetaMap__.getKey(), entryMetaMap__.getValue());
            break;
          }
          case 16: {

            embeddingSize_ = input.readUInt32();
            break;
          }
          case 26: {
            java.lang.String s = input.readStringRequireUtf8();

            model_ = s;
            break;
          }
        }
      }
    } catch (com.google.protobuf.InvalidProtocolBufferException e) {
      throw e.setUnfinishedMessage(this);
    } catch (java.io.IOException e) {
      throw new com.google.protobuf.InvalidProtocolBufferException(
          e).setUnfinishedMessage(this);
    } finally {
      this.unknownFields = unknownFields.build();
      makeExtensionsImmutable();
    }
  }
  public static final com.google.protobuf.Descriptors.Descriptor
      getDescriptor() {
    return org.openmit.entry.EntryMetaProtos.internal_static_mit_protobuf_EntryMeta_descriptor;
  }

  @SuppressWarnings({"rawtypes"})
  protected com.google.protobuf.MapField internalGetMapField(
      int number) {
    switch (number) {
      case 1:
        return internalGetEntryMetaMap();
      default:
        throw new RuntimeException(
            "Invalid map field number: " + number);
    }
  }
  protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
      internalGetFieldAccessorTable() {
    return org.openmit.entry.EntryMetaProtos.internal_static_mit_protobuf_EntryMeta_fieldAccessorTable
        .ensureFieldAccessorsInitialized(
            org.openmit.entry.EntryMeta.class, org.openmit.entry.EntryMeta.Builder.class);
  }

  private int bitField0_;
  public static final int ENTRY_META_MAP_FIELD_NUMBER = 1;
  private static final class EntryMetaMapDefaultEntryHolder {
    static final com.google.protobuf.MapEntry<
        java.lang.String, org.openmit.entry.FieldIdArray> defaultEntry =
            com.google.protobuf.MapEntry
            .<java.lang.String, org.openmit.entry.FieldIdArray>newDefaultInstance(
                org.openmit.entry.EntryMetaProtos.internal_static_mit_protobuf_EntryMeta_EntryMetaMapEntry_descriptor, 
                com.google.protobuf.WireFormat.FieldType.STRING,
                "",
                com.google.protobuf.WireFormat.FieldType.MESSAGE,
                org.openmit.entry.FieldIdArray.getDefaultInstance());
  }
  private com.google.protobuf.MapField<
      java.lang.String, org.openmit.entry.FieldIdArray> entryMetaMap_;
  private com.google.protobuf.MapField<java.lang.String, org.openmit.entry.FieldIdArray>
  internalGetEntryMetaMap() {
    if (entryMetaMap_ == null) {
      return com.google.protobuf.MapField.emptyMapField(
          EntryMetaMapDefaultEntryHolder.defaultEntry);
    }
    return entryMetaMap_;
  }

  public int getEntryMetaMapCount() {
    return internalGetEntryMetaMap().getMap().size();
  }
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */

  public boolean containsEntryMetaMap(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    return internalGetEntryMetaMap().getMap().containsKey(key);
  }
  /**
   * Use {@link #getEntryMetaMapMap()} instead.
   */
  @java.lang.Deprecated
  public java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> getEntryMetaMap() {
    return getEntryMetaMapMap();
  }
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */

  public java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> getEntryMetaMapMap() {
    return internalGetEntryMetaMap().getMap();
  }
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */

  public org.openmit.entry.FieldIdArray getEntryMetaMapOrDefault(
      java.lang.String key,
      org.openmit.entry.FieldIdArray defaultValue) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> map =
        internalGetEntryMetaMap().getMap();
    return map.containsKey(key) ? map.get(key) : defaultValue;
  }
  /**
   * <pre>
   * Maps of between fieldid name and related fieldid list
   * </pre>
   *
   * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
   */

  public org.openmit.entry.FieldIdArray getEntryMetaMapOrThrow(
      java.lang.String key) {
    if (key == null) { throw new java.lang.NullPointerException(); }
    java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> map =
        internalGetEntryMetaMap().getMap();
    if (!map.containsKey(key)) {
      throw new java.lang.IllegalArgumentException();
    }
    return map.get(key);
  }

  public static final int EMBEDDING_SIZE_FIELD_NUMBER = 2;
  private int embeddingSize_;
  /**
   * <pre>
   * embedding size 
   * </pre>
   *
   * <code>uint32 embedding_size = 2;</code>
   */
  public int getEmbeddingSize() {
    return embeddingSize_;
  }

  public static final int MODEL_FIELD_NUMBER = 3;
  private volatile java.lang.Object model_;
  /**
   * <pre>
   * model name 
   * </pre>
   *
   * <code>string model = 3;</code>
   */
  public java.lang.String getModel() {
    java.lang.Object ref = model_;
    if (ref instanceof java.lang.String) {
      return (java.lang.String) ref;
    } else {
      com.google.protobuf.ByteString bs = 
          (com.google.protobuf.ByteString) ref;
      java.lang.String s = bs.toStringUtf8();
      model_ = s;
      return s;
    }
  }
  /**
   * <pre>
   * model name 
   * </pre>
   *
   * <code>string model = 3;</code>
   */
  public com.google.protobuf.ByteString
      getModelBytes() {
    java.lang.Object ref = model_;
    if (ref instanceof java.lang.String) {
      com.google.protobuf.ByteString b = 
          com.google.protobuf.ByteString.copyFromUtf8(
              (java.lang.String) ref);
      model_ = b;
      return b;
    } else {
      return (com.google.protobuf.ByteString) ref;
    }
  }

  private byte memoizedIsInitialized = -1;
  public final boolean isInitialized() {
    byte isInitialized = memoizedIsInitialized;
    if (isInitialized == 1) return true;
    if (isInitialized == 0) return false;

    memoizedIsInitialized = 1;
    return true;
  }

  public void writeTo(com.google.protobuf.CodedOutputStream output)
                      throws java.io.IOException {
    com.google.protobuf.GeneratedMessageV3
      .serializeStringMapTo(
        output,
        internalGetEntryMetaMap(),
        EntryMetaMapDefaultEntryHolder.defaultEntry,
        1);
    if (embeddingSize_ != 0) {
      output.writeUInt32(2, embeddingSize_);
    }
    if (!getModelBytes().isEmpty()) {
      com.google.protobuf.GeneratedMessageV3.writeString(output, 3, model_);
    }
    unknownFields.writeTo(output);
  }

  public int getSerializedSize() {
    int size = memoizedSize;
    if (size != -1) return size;

    size = 0;
    for (java.util.Map.Entry<java.lang.String, org.openmit.entry.FieldIdArray> entry
         : internalGetEntryMetaMap().getMap().entrySet()) {
      com.google.protobuf.MapEntry<java.lang.String, org.openmit.entry.FieldIdArray>
      entryMetaMap__ = EntryMetaMapDefaultEntryHolder.defaultEntry.newBuilderForType()
          .setKey(entry.getKey())
          .setValue(entry.getValue())
          .build();
      size += com.google.protobuf.CodedOutputStream
          .computeMessageSize(1, entryMetaMap__);
    }
    if (embeddingSize_ != 0) {
      size += com.google.protobuf.CodedOutputStream
        .computeUInt32Size(2, embeddingSize_);
    }
    if (!getModelBytes().isEmpty()) {
      size += com.google.protobuf.GeneratedMessageV3.computeStringSize(3, model_);
    }
    size += unknownFields.getSerializedSize();
    memoizedSize = size;
    return size;
  }

  @java.lang.Override
  public boolean equals(final java.lang.Object obj) {
    if (obj == this) {
     return true;
    }
    if (!(obj instanceof org.openmit.entry.EntryMeta)) {
      return super.equals(obj);
    }
    org.openmit.entry.EntryMeta other = (org.openmit.entry.EntryMeta) obj;

    boolean result = true;
    result = result && internalGetEntryMetaMap().equals(
        other.internalGetEntryMetaMap());
    result = result && (getEmbeddingSize()
        == other.getEmbeddingSize());
    result = result && getModel()
        .equals(other.getModel());
    result = result && unknownFields.equals(other.unknownFields);
    return result;
  }

  @java.lang.Override
  public int hashCode() {
    if (memoizedHashCode != 0) {
      return memoizedHashCode;
    }
    int hash = 41;
    hash = (19 * hash) + getDescriptor().hashCode();
    if (!internalGetEntryMetaMap().getMap().isEmpty()) {
      hash = (37 * hash) + ENTRY_META_MAP_FIELD_NUMBER;
      hash = (53 * hash) + internalGetEntryMetaMap().hashCode();
    }
    hash = (37 * hash) + EMBEDDING_SIZE_FIELD_NUMBER;
    hash = (53 * hash) + getEmbeddingSize();
    hash = (37 * hash) + MODEL_FIELD_NUMBER;
    hash = (53 * hash) + getModel().hashCode();
    hash = (29 * hash) + unknownFields.hashCode();
    memoizedHashCode = hash;
    return hash;
  }

  public static org.openmit.entry.EntryMeta parseFrom(
      java.nio.ByteBuffer data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      java.nio.ByteBuffer data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      com.google.protobuf.ByteString data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      com.google.protobuf.ByteString data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.openmit.entry.EntryMeta parseFrom(byte[] data)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      byte[] data,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws com.google.protobuf.InvalidProtocolBufferException {
    return PARSER.parseFrom(data, extensionRegistry);
  }
  public static org.openmit.entry.EntryMeta parseFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.openmit.entry.EntryMeta parseDelimitedFrom(java.io.InputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input);
  }
  public static org.openmit.entry.EntryMeta parseDelimitedFrom(
      java.io.InputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseDelimitedWithIOException(PARSER, input, extensionRegistry);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      com.google.protobuf.CodedInputStream input)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input);
  }
  public static org.openmit.entry.EntryMeta parseFrom(
      com.google.protobuf.CodedInputStream input,
      com.google.protobuf.ExtensionRegistryLite extensionRegistry)
      throws java.io.IOException {
    return com.google.protobuf.GeneratedMessageV3
        .parseWithIOException(PARSER, input, extensionRegistry);
  }

  public Builder newBuilderForType() { return newBuilder(); }
  public static Builder newBuilder() {
    return DEFAULT_INSTANCE.toBuilder();
  }
  public static Builder newBuilder(org.openmit.entry.EntryMeta prototype) {
    return DEFAULT_INSTANCE.toBuilder().mergeFrom(prototype);
  }
  public Builder toBuilder() {
    return this == DEFAULT_INSTANCE
        ? new Builder() : new Builder().mergeFrom(this);
  }

  @java.lang.Override
  protected Builder newBuilderForType(
      com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
    Builder builder = new Builder(parent);
    return builder;
  }
  /**
   * Protobuf type {@code mit.protobuf.EntryMeta}
   */
  public static final class Builder extends
      com.google.protobuf.GeneratedMessageV3.Builder<Builder> implements
      // @@protoc_insertion_point(builder_implements:mit.protobuf.EntryMeta)
      org.openmit.entry.EntryMetaOrBuilder {
    public static final com.google.protobuf.Descriptors.Descriptor
        getDescriptor() {
      return org.openmit.entry.EntryMetaProtos.internal_static_mit_protobuf_EntryMeta_descriptor;
    }

    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetEntryMetaMap();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    @SuppressWarnings({"rawtypes"})
    protected com.google.protobuf.MapField internalGetMutableMapField(
        int number) {
      switch (number) {
        case 1:
          return internalGetMutableEntryMetaMap();
        default:
          throw new RuntimeException(
              "Invalid map field number: " + number);
      }
    }
    protected com.google.protobuf.GeneratedMessageV3.FieldAccessorTable
        internalGetFieldAccessorTable() {
      return org.openmit.entry.EntryMetaProtos.internal_static_mit_protobuf_EntryMeta_fieldAccessorTable
          .ensureFieldAccessorsInitialized(
              org.openmit.entry.EntryMeta.class, org.openmit.entry.EntryMeta.Builder.class);
    }

    // Construct using org.openmit.entry.EntryMeta.newBuilder()
    private Builder() {
      maybeForceBuilderInitialization();
    }

    private Builder(
        com.google.protobuf.GeneratedMessageV3.BuilderParent parent) {
      super(parent);
      maybeForceBuilderInitialization();
    }
    private void maybeForceBuilderInitialization() {
      if (com.google.protobuf.GeneratedMessageV3
              .alwaysUseFieldBuilders) {
      }
    }
    public Builder clear() {
      super.clear();
      internalGetMutableEntryMetaMap().clear();
      embeddingSize_ = 0;

      model_ = "";

      return this;
    }

    public com.google.protobuf.Descriptors.Descriptor
        getDescriptorForType() {
      return org.openmit.entry.EntryMetaProtos.internal_static_mit_protobuf_EntryMeta_descriptor;
    }

    public org.openmit.entry.EntryMeta getDefaultInstanceForType() {
      return org.openmit.entry.EntryMeta.getDefaultInstance();
    }

    public org.openmit.entry.EntryMeta build() {
      org.openmit.entry.EntryMeta result = buildPartial();
      if (!result.isInitialized()) {
        throw newUninitializedMessageException(result);
      }
      return result;
    }

    public org.openmit.entry.EntryMeta buildPartial() {
      org.openmit.entry.EntryMeta result = new org.openmit.entry.EntryMeta(this);
      int from_bitField0_ = bitField0_;
      int to_bitField0_ = 0;
      result.entryMetaMap_ = internalGetEntryMetaMap();
      result.entryMetaMap_.makeImmutable();
      result.embeddingSize_ = embeddingSize_;
      result.model_ = model_;
      result.bitField0_ = to_bitField0_;
      onBuilt();
      return result;
    }

    public Builder clone() {
      return (Builder) super.clone();
    }
    public Builder setField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.setField(field, value);
    }
    public Builder clearField(
        com.google.protobuf.Descriptors.FieldDescriptor field) {
      return (Builder) super.clearField(field);
    }
    public Builder clearOneof(
        com.google.protobuf.Descriptors.OneofDescriptor oneof) {
      return (Builder) super.clearOneof(oneof);
    }
    public Builder setRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        int index, java.lang.Object value) {
      return (Builder) super.setRepeatedField(field, index, value);
    }
    public Builder addRepeatedField(
        com.google.protobuf.Descriptors.FieldDescriptor field,
        java.lang.Object value) {
      return (Builder) super.addRepeatedField(field, value);
    }
    public Builder mergeFrom(com.google.protobuf.Message other) {
      if (other instanceof org.openmit.entry.EntryMeta) {
        return mergeFrom((org.openmit.entry.EntryMeta)other);
      } else {
        super.mergeFrom(other);
        return this;
      }
    }

    public Builder mergeFrom(org.openmit.entry.EntryMeta other) {
      if (other == org.openmit.entry.EntryMeta.getDefaultInstance()) return this;
      internalGetMutableEntryMetaMap().mergeFrom(
          other.internalGetEntryMetaMap());
      if (other.getEmbeddingSize() != 0) {
        setEmbeddingSize(other.getEmbeddingSize());
      }
      if (!other.getModel().isEmpty()) {
        model_ = other.model_;
        onChanged();
      }
      this.mergeUnknownFields(other.unknownFields);
      onChanged();
      return this;
    }

    public final boolean isInitialized() {
      return true;
    }

    public Builder mergeFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws java.io.IOException {
      org.openmit.entry.EntryMeta parsedMessage = null;
      try {
        parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
      } catch (com.google.protobuf.InvalidProtocolBufferException e) {
        parsedMessage = (org.openmit.entry.EntryMeta) e.getUnfinishedMessage();
        throw e.unwrapIOException();
      } finally {
        if (parsedMessage != null) {
          mergeFrom(parsedMessage);
        }
      }
      return this;
    }
    private int bitField0_;

    private com.google.protobuf.MapField<
        java.lang.String, org.openmit.entry.FieldIdArray> entryMetaMap_;
    private com.google.protobuf.MapField<java.lang.String, org.openmit.entry.FieldIdArray>
    internalGetEntryMetaMap() {
      if (entryMetaMap_ == null) {
        return com.google.protobuf.MapField.emptyMapField(
            EntryMetaMapDefaultEntryHolder.defaultEntry);
      }
      return entryMetaMap_;
    }
    private com.google.protobuf.MapField<java.lang.String, org.openmit.entry.FieldIdArray>
    internalGetMutableEntryMetaMap() {
      onChanged();;
      if (entryMetaMap_ == null) {
        entryMetaMap_ = com.google.protobuf.MapField.newMapField(
            EntryMetaMapDefaultEntryHolder.defaultEntry);
      }
      if (!entryMetaMap_.isMutable()) {
        entryMetaMap_ = entryMetaMap_.copy();
      }
      return entryMetaMap_;
    }

    public int getEntryMetaMapCount() {
      return internalGetEntryMetaMap().getMap().size();
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */

    public boolean containsEntryMetaMap(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      return internalGetEntryMetaMap().getMap().containsKey(key);
    }
    /**
     * Use {@link #getEntryMetaMapMap()} instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> getEntryMetaMap() {
      return getEntryMetaMapMap();
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */

    public java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> getEntryMetaMapMap() {
      return internalGetEntryMetaMap().getMap();
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */

    public org.openmit.entry.FieldIdArray getEntryMetaMapOrDefault(
        java.lang.String key,
        org.openmit.entry.FieldIdArray defaultValue) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> map =
          internalGetEntryMetaMap().getMap();
      return map.containsKey(key) ? map.get(key) : defaultValue;
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */

    public org.openmit.entry.FieldIdArray getEntryMetaMapOrThrow(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> map =
          internalGetEntryMetaMap().getMap();
      if (!map.containsKey(key)) {
        throw new java.lang.IllegalArgumentException();
      }
      return map.get(key);
    }

    public Builder clearEntryMetaMap() {
      internalGetMutableEntryMetaMap().getMutableMap()
          .clear();
      return this;
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */

    public Builder removeEntryMetaMap(
        java.lang.String key) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      internalGetMutableEntryMetaMap().getMutableMap()
          .remove(key);
      return this;
    }
    /**
     * Use alternate mutation accessors instead.
     */
    @java.lang.Deprecated
    public java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray>
    getMutableEntryMetaMap() {
      return internalGetMutableEntryMetaMap().getMutableMap();
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */
    public Builder putEntryMetaMap(
        java.lang.String key,
        org.openmit.entry.FieldIdArray value) {
      if (key == null) { throw new java.lang.NullPointerException(); }
      if (value == null) { throw new java.lang.NullPointerException(); }
      internalGetMutableEntryMetaMap().getMutableMap()
          .put(key, value);
      return this;
    }
    /**
     * <pre>
     * Maps of between fieldid name and related fieldid list
     * </pre>
     *
     * <code>map&lt;string, .mit.protobuf.FieldIdArray&gt; entry_meta_map = 1;</code>
     */

    public Builder putAllEntryMetaMap(
        java.util.Map<java.lang.String, org.openmit.entry.FieldIdArray> values) {
      internalGetMutableEntryMetaMap().getMutableMap()
          .putAll(values);
      return this;
    }

    private int embeddingSize_ ;
    /**
     * <pre>
     * embedding size 
     * </pre>
     *
     * <code>uint32 embedding_size = 2;</code>
     */
    public int getEmbeddingSize() {
      return embeddingSize_;
    }
    /**
     * <pre>
     * embedding size 
     * </pre>
     *
     * <code>uint32 embedding_size = 2;</code>
     */
    public Builder setEmbeddingSize(int value) {
      
      embeddingSize_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * embedding size 
     * </pre>
     *
     * <code>uint32 embedding_size = 2;</code>
     */
    public Builder clearEmbeddingSize() {
      
      embeddingSize_ = 0;
      onChanged();
      return this;
    }

    private java.lang.Object model_ = "";
    /**
     * <pre>
     * model name 
     * </pre>
     *
     * <code>string model = 3;</code>
     */
    public java.lang.String getModel() {
      java.lang.Object ref = model_;
      if (!(ref instanceof java.lang.String)) {
        com.google.protobuf.ByteString bs =
            (com.google.protobuf.ByteString) ref;
        java.lang.String s = bs.toStringUtf8();
        model_ = s;
        return s;
      } else {
        return (java.lang.String) ref;
      }
    }
    /**
     * <pre>
     * model name 
     * </pre>
     *
     * <code>string model = 3;</code>
     */
    public com.google.protobuf.ByteString
        getModelBytes() {
      java.lang.Object ref = model_;
      if (ref instanceof String) {
        com.google.protobuf.ByteString b = 
            com.google.protobuf.ByteString.copyFromUtf8(
                (java.lang.String) ref);
        model_ = b;
        return b;
      } else {
        return (com.google.protobuf.ByteString) ref;
      }
    }
    /**
     * <pre>
     * model name 
     * </pre>
     *
     * <code>string model = 3;</code>
     */
    public Builder setModel(
        java.lang.String value) {
      if (value == null) {
    throw new NullPointerException();
  }
  
      model_ = value;
      onChanged();
      return this;
    }
    /**
     * <pre>
     * model name 
     * </pre>
     *
     * <code>string model = 3;</code>
     */
    public Builder clearModel() {
      
      model_ = getDefaultInstance().getModel();
      onChanged();
      return this;
    }
    /**
     * <pre>
     * model name 
     * </pre>
     *
     * <code>string model = 3;</code>
     */
    public Builder setModelBytes(
        com.google.protobuf.ByteString value) {
      if (value == null) {
    throw new NullPointerException();
  }
  checkByteStringIsUtf8(value);
      
      model_ = value;
      onChanged();
      return this;
    }
    public final Builder setUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.setUnknownFieldsProto3(unknownFields);
    }

    public final Builder mergeUnknownFields(
        final com.google.protobuf.UnknownFieldSet unknownFields) {
      return super.mergeUnknownFields(unknownFields);
    }


    // @@protoc_insertion_point(builder_scope:mit.protobuf.EntryMeta)
  }

  // @@protoc_insertion_point(class_scope:mit.protobuf.EntryMeta)
  private static final org.openmit.entry.EntryMeta DEFAULT_INSTANCE;
  static {
    DEFAULT_INSTANCE = new org.openmit.entry.EntryMeta();
  }

  public static org.openmit.entry.EntryMeta getDefaultInstance() {
    return DEFAULT_INSTANCE;
  }

  private static final com.google.protobuf.Parser<EntryMeta>
      PARSER = new com.google.protobuf.AbstractParser<EntryMeta>() {
    public EntryMeta parsePartialFrom(
        com.google.protobuf.CodedInputStream input,
        com.google.protobuf.ExtensionRegistryLite extensionRegistry)
        throws com.google.protobuf.InvalidProtocolBufferException {
      return new EntryMeta(input, extensionRegistry);
    }
  };

  public static com.google.protobuf.Parser<EntryMeta> parser() {
    return PARSER;
  }

  @java.lang.Override
  public com.google.protobuf.Parser<EntryMeta> getParserForType() {
    return PARSER;
  }

  public org.openmit.entry.EntryMeta getDefaultInstanceForType() {
    return DEFAULT_INSTANCE;
  }

}

