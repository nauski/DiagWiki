import os

class RepoUtil:
    """Utility class for repository operations"""

    @staticmethod
    def build_tree(current_path):
        # ignore hidden files and folders
        if os.path.basename(current_path).startswith('.'):
            return None
        # ignore __pycache__ folders
        if os.path.basename(current_path) == '__pycache__':
            return None
        if os.path.isdir(current_path):
            children = []
            for item in os.listdir(current_path):
                item_path = os.path.join(current_path, item)
                child_tree = RepoUtil.build_tree(item_path)
                if child_tree is not None:
                    children.append(child_tree)
            return {
                "name": os.path.basename(current_path),
                "type": "directory",
                "children": children
            }
        else:
            return {
                "name": os.path.basename(current_path),
                "type": "file"
            }