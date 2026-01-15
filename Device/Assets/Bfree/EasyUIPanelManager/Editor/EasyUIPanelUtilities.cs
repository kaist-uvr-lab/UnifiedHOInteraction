using UnityEditor;
using UnityEngine;

namespace Bfree
{
    public class EasyUIPanelUtilities : EditorWindow
    {

        [MenuItem("GameObject/EasyUIPanel/PanelManager", false, 1)]
        static void CreatePanelManager()
        {
            //Parent
            GameObject prefab = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<Object>("Assets/Bfree/EasyUIPanelManager/Prefabs/PanelManager.prefab"));
            prefab.name = "PanelManager";

            if (Selection.activeTransform != null)
            {
                prefab.transform.SetParent(Selection.activeTransform, false);
            }
            prefab.transform.localPosition = Vector3.zero;
            prefab.transform.localEulerAngles = Vector3.zero;
            prefab.transform.localScale = Vector3.one;
        }
        [MenuItem("GameObject/EasyUIPanel/Sample Canvases", false, 2)]
        static void CreateSampleCanvases()
        {
            //Parent
            GameObject prefab = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<Object>("Assets/Bfree/EasyUIPanelManager/Prefabs/Sample_canvases.prefab"));
            prefab.name = "Sample Canvases";

            if (Selection.activeTransform != null)
            {
                prefab.transform.SetParent(Selection.activeTransform, false);
            }
            prefab.transform.localPosition = Vector3.zero;
            prefab.transform.localEulerAngles = Vector3.zero;
            prefab.transform.localScale = Vector3.one;
        }
        [MenuItem("GameObject/EasyUIPanel/Tab System", false, 3)]
        static void CreateTabs()
        {
            //Parent
            GameObject prefab = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<Object>("Assets/Bfree/EasyUIPanelManager/Prefabs/UIElements/Tabs.prefab"));
            prefab.name = "Tab System";

            if (Selection.activeTransform != null)
            {
                prefab.transform.SetParent(Selection.activeTransform, false);
            }
            prefab.transform.localPosition = Vector3.zero;
            prefab.transform.localEulerAngles = Vector3.zero;
            prefab.transform.localScale = Vector3.one;
        }
        [MenuItem("GameObject/EasyUIPanel/Nested Panel", false, 4)]
        static void CreateNestedPanel()
        {
            //Parent
            GameObject prefab = (GameObject)PrefabUtility.InstantiatePrefab(AssetDatabase.LoadAssetAtPath<Object>("Assets/Bfree/EasyUIPanelManager/Prefabs/UIElements/Nested Panel.prefab"));
            prefab.name = "Nested Panel";

            if (Selection.activeTransform != null)
            {
                prefab.transform.SetParent(Selection.activeTransform, false);
            }
            prefab.transform.localPosition = Vector3.zero;
            prefab.transform.localEulerAngles = Vector3.zero;
            prefab.transform.localScale = Vector3.one;
        }
    }
}