# prefabConvert

prefabConvert takes .prefab (and required .meta files) to convert; as it's designed for AssetRipper outputs, it will therefore also need a Unity-style Assets/ project structure (will automatically detect upwards). It will by default export OBJ+MTL; if you want .FBX outputs then pass `--fbx` and it'll use the blender backend to convert the default outputs (obj/mtl) into an fbx file. 

```sh
uv run prefabconvert \
  --input "/path/to/prefabs" \
  --output "/path/to/output" \
  --recursive
```

If Blender is not auto-detected, pass its absolute binary path w/ `--blender-path`, e.g. `--blender-path /Applications/Blender.app/Contents/MacOS/Blender` or `--blender-path C:\Program Files\Blender Foundation\Blender [Version]\blender.exe`

> This software is not sponsored by or affiliated with Unity Technologies or its affiliates. "Unity" is a registered trademark of Unity Technologies or its affiliates in the U.S. and elsewhere.