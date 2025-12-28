from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import PyPDF2
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")

# NLP библиотеки


class PlagiarismDetector:
    """Основной класс детектора плагиата"""

    def __init__(self, min_similarity_threshold: float = 0.3, language: str = "english"):
        """
        Инициализация детектора плагиата

        Args:
            min_similarity_threshold: минимальный порог схожести для отчета
            language: язык текстов ('english' или 'russian')
        """
        self.documents: List[Document] = []
        self.min_threshold = min_similarity_threshold
        self.language = language

        # Инициализация NLP компонентов
        self._initialize_nlp_components()

    def _initialize_nlp_components(self) -> None:
        """Инициализация и загрузка необходимых NLP компонентов"""
        print("⚙️ Инициализация NLP компонентов...")

        try:
            # Скачивание необходимых ресурсов NLTK
            required_packages = ["punkt", "wordnet", "stopwords", "punkt_tab"]

            for package in required_packages:
                try:
                    nltk.data.find(
                        f"tokenizers/{package}" if package == "punkt_tab" else package
                    )
                except LookupError:
                    print(f"  📥 Загрузка {package}...")
                    nltk.download(package, quiet=True)

            # Инициализация лемматизатора и стоп-слов
            if self.language == "russian":
                # Для русского языка используем SnowballStemmer вместо WordNetLemmatizer
                from nltk.stem import SnowballStemmer

                self.lemmatizer = SnowballStemmer("russian")
                # Простой список стоп-слов для русского
                self.stop_words = {
                    "и",
                    "в",
                    "во",
                    "не",
                    "что",
                    "он",
                    "на",
                    "я",
                    "с",
                    "со",
                    "как",
                    "а",
                    "то",
                    "все",
                    "она",
                    "так",
                    "его",
                    "но",
                    "да",
                    "ты",
                    "к",
                    "у",
                    "же",
                    "вы",
                    "за",
                    "бы",
                    "по",
                    "только",
                    "ее",
                    "мне",
                    "было",
                    "вот",
                    "от",
                    "меня",
                    "еще",
                    "нет",
                    "о",
                    "из",
                    "ему",
                    "теперь",
                    "когда",
                    "даже",
                    "ну",
                    "вдруг",
                    "ли",
                    "если",
                    "уже",
                    "или",
                    "ни",
                    "быть",
                    "был",
                    "него",
                    "до",
                    "вас",
                    "нибудь",
                    "опять",
                    "уж",
                    "вам",
                    "ведь",
                    "там",
                    "потом",
                    "себя",
                    "ничего",
                    "ей",
                    "может",
                    "они",
                    "тут",
                    "где",
                    "есть",
                    "надо",
                    "ней",
                    "для",
                    "мы",
                    "тебя",
                    "их",
                    "чем",
                    "была",
                    "сам",
                    "чтоб",
                    "без",
                    "будто",
                    "чего",
                    "раз",
                    "тоже",
                    "себе",
                    "под",
                    "будет",
                    "ж",
                    "тогда",
                    "кто",
                    "этот",
                    "того",
                    "потому",
                    "этого",
                    "какой",
                    "совсем",
                    "ним",
                    "здесь",
                    "этом",
                    "один",
                    "почти",
                    "мой",
                    "тем",
                    "чтобы",
                    "нее",
                    "сейчас",
                    "были",
                    "куда",
                    "зачем",
                    "всех",
                    "никогда",
                    "можно",
                    "при",
                    "наконец",
                    "два",
                    "об",
                    "другой",
                    "хоть",
                    "после",
                    "над",
                    "больше",
                    "тот",
                    "через",
                    "эти",
                    "нас",
                    "про",
                    "всего",
                    "них",
                    "какая",
                    "много",
                    "разве",
                    "три",
                    "эту",
                    "моя",
                    "впрочем",
                    "хорошо",
                    "свою",
                    "этой",
                    "перед",
                    "иногда",
                    "лучше",
                    "чуть",
                    "том",
                    "нельзя",
                    "такой",
                    "им",
                    "более",
                    "всегда",
                    "конечно",
                    "всю",
                    "между",
                }
            else:
                # Для английского языка
                self.lemmatizer = WordNetLemmatizer()
                self.stop_words = set(stopwords.words("english"))

            print("✓ NLP компоненты инициализированы")

        except Exception as e:
            print(f"❌ Ошибка при инициализации NLP компонентов: {str(e)}")
            print("Попытка использовать упрощенную обработку...")
            self.lemmatizer = None
            self.stop_words = set()

    @dataclass
    class Document:
        """Класс для представления документа"""

        filename: str
        content: str
        processed_content: str = ""
        file_type: str = ""

    def load_documents(self, folder_path: str) -> None:
        """
        Загрузка документов из папки

        Args:
            folder_path: путь к папке с документами
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Папка {folder_path} не найдена")

        supported_extensions = {".txt", ".pdf"}

        for file_path in folder.iterdir():
            if file_path.suffix.lower() in supported_extensions:
                try:
                    if file_path.suffix.lower() == ".txt":
                        content = self._read_txt_file(file_path)
                        file_type = "txt"
                    elif file_path.suffix.lower() == ".pdf":
                        content = self._read_pdf_file(file_path)
                        file_type = "pdf"
                    else:
                        continue

                    doc = self.Document(
                        filename=file_path.name, content=content, file_type=file_type
                    )
                    self.documents.append(doc)
                    print(f"✓ Загружен: {file_path.name} ({len(content)} символов)")

                except Exception as e:
                    print(f"✗ Ошибка при чтении {file_path.name}: {str(e)}")

    def _read_txt_file(self, file_path: Path) -> str:
        """Чтение текстового файла"""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            # Попытка с другой кодировкой
            with open(file_path, "r", encoding="cp1251") as file:
                return file.read()

    def _read_pdf_file(self, file_path: Path) -> str:
        """Чтение PDF файла"""
        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"  ⚠️ Предупреждение при чтении PDF: {str(e)}")
        return text

    def preprocess_text(self, text: str) -> str:
        """
        Предварительная обработка текста

        Args:
            text: исходный текст

        Returns:
            Обработанный текст
        """
        if not text or not text.strip():
            return ""

        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление специальных символов и цифр
        if self.language == "russian":
            # Для русского языка сохраняем кириллицу
            text = re.sub(r"[^а-яё\s]", " ", text, flags=re.IGNORECASE)
        else:
            # Для английского языка
            text = re.sub(r"[^a-z\s]", " ", text)

        # Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()

        # Упрощенная токенизация (без NLTK если есть проблемы)
        try:
            tokens = word_tokenize(text, language=self.language)
        except BaseException:
            # Резервный метод токенизации
            tokens = text.split()

        # Лемматизация и удаление стоп-слов
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                if self.lemmatizer:
                    try:
                        if hasattr(self.lemmatizer, "lemmatize"):
                            lemma = self.lemmatizer.lemmatize(token)
                        else:
                            # Для SnowballStemmer
                            lemma = self.lemmatizer.stem(token)
                        processed_tokens.append(lemma)
                    except BaseException:
                        processed_tokens.append(token)
                else:
                    processed_tokens.append(token)

        return " ".join(processed_tokens)

    def process_all_documents(self) -> None:
        """Обработка всех загруженных документов"""
        print("\n⏳ Обработка документов...")
        for i, doc in enumerate(self.documents):
            try:
                original_length = len(doc.content)
                doc.processed_content = self.preprocess_text(doc.content)
                processed_length = len(doc.processed_content.split())
                print(
                    f"  {i+1}. {doc.filename}: {original_length} симв. → {processed_length} слов"
                )
            except Exception as e:
                print(f"  ✗ Ошибка при обработке {doc.filename}: {str(e)}")
                doc.processed_content = ""
        print("✓ Обработка завершена")

    def cosine_similarity_method(self, text1: str, text2: str) -> float:
        """
        Сравнение текстов методом косинусного сходства с TF-IDF

        Args:
            text1: первый текст
            text2: второй текст

        Returns:
            Коэффициент схожести от 0 до 1
        """
        if not text1.strip() or not text2.strip():
            return 0.0

        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"  ⚠️ Ошибка в cosine_similarity: {str(e)}")
            return 0.0

    def longest_common_subsequence(self, text1: str, text2: str) -> float:
        """
        Поиск самой длинной общей подпоследовательности

        Args:
            text1: первый текст
            text2: второй текст

        Returns:
            Нормализованный коэффициент схожести
        """
        words1 = text1.split()
        words2 = text2.split()

        if not words1 or not words2:
            return 0.0

        m, n = len(words1), len(words2)

        # Создание матрицы для динамического программирования
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Заполнение матрицы
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i - 1] == words2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]

        # Нормализация по длине самого короткого текста
        min_length = min(m, n)
        if min_length == 0:
            return 0.0

        return lcs_length / min_length

    def ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """
        Сравнение текстов с использованием N-gram

        Args:
            text1: первый текст
            text2: второй текст
            n: размер N-gram

        Returns:
            Коэффициент схожести Jaccard для N-gram
        """

        def get_ngrams(text, n):
            words = text.split()
            if len(words) < n:
                return set()
            ngrams = set()
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i: i + n])
                ngrams.add(ngram)
            return ngrams

        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))

        return intersection / union if union > 0 else 0.0

    def calculate_similarity_matrix(self) -> Dict[str, np.ndarray]:
        """
        Расчет матриц схожести всеми методами

        Returns:
            Словарь с матрицами схожести для каждого метода
        """
        n = len(self.documents)
        if n == 0:
            return {}

        filenames = [doc.filename for doc in self.documents]

        # Инициализация матриц
        cosine_matrix = np.zeros((n, n))
        lcs_matrix = np.zeros((n, n))
        ngram_matrix = np.zeros((n, n))
        combined_matrix = np.zeros((n, n))

        print(f"\n🧮 Расчет схожести для {n} документов...")

        for i in range(n):
            for j in range(i, n):
                text1 = self.documents[i].processed_content
                text2 = self.documents[j].processed_content

                # Расчет всеми методами
                cosine_sim = self.cosine_similarity_method(text1, text2)
                lcs_sim = self.longest_common_subsequence(text1, text2)
                ngram_sim = self.ngram_similarity(text1, text2, n=3)

                # Комбинированная оценка (среднее взвешенное)
                combined = 0.4 * cosine_sim + 0.3 * lcs_sim + 0.3 * ngram_sim

                cosine_matrix[i, j] = cosine_sim
                cosine_matrix[j, i] = cosine_sim

                lcs_matrix[i, j] = lcs_sim
                lcs_matrix[j, i] = lcs_sim

                ngram_matrix[i, j] = ngram_sim
                ngram_matrix[j, i] = ngram_sim

                combined_matrix[i, j] = combined
                combined_matrix[j, i] = combined

        print("✓ Расчет схожести завершен")

        return {
            "cosine": cosine_matrix,
            "lcs": lcs_matrix,
            "ngram": ngram_matrix,
            "combined": combined_matrix,
            "filenames": filenames,
        }

    def visualize_results(self, similarity_matrices: Dict[str, np.ndarray]) -> None:
        """
        Визуализация результатов в виде тепловых карт

        Args:
            similarity_matrices: словарь с матрицами схожести
        """
        if not similarity_matrices:
            print("❌ Нет данных для визуализации")
            return

        filenames = similarity_matrices["filenames"]
        if len(filenames) < 2:
            print("⚠️ Недостаточно документов для визуализации")
            return

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            methods = ["cosine", "lcs", "ngram", "combined"]
            titles = [
                "Cosine Similarity",
                "LCS Similarity",
                "N-gram Similarity",
                "Combined Similarity",
            ]

            for idx, (method, title) in enumerate(zip(methods, titles)):
                ax = axes[idx]
                matrix = similarity_matrices[method]

                # Создание тепловой карты
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt=".2f",
                    cmap="RdYlGn_r",
                    square=True,
                    ax=ax,
                    xticklabels=filenames,
                    yticklabels=filenames,
                    cbar_kws={"label": "Similarity Score"},
                    vmin=0,
                    vmax=1,
                )

                ax.set_title(title, fontsize=14, fontweight="bold")
                ax.set_xlabel("Documents")
                ax.set_ylabel("Documents")
                ax.tick_params(axis="x", rotation=45)
                ax.tick_params(axis="y", rotation=0)

            plt.tight_layout()
            plt.savefig("similarity_matrix.png", dpi=300, bbox_inches="tight")
            print("✓ Визуализация сохранена в similarity_matrix.png")
            plt.show()

        except Exception as e:
            print(f"⚠️ Ошибка при визуализации: {str(e)}")
            # Простая текстовая визуализация
            self._text_visualization(similarity_matrices)

        # Сохранение матриц в файл
        self._save_matrices_to_csv(similarity_matrices)

    def _text_visualization(self, similarity_matrices: Dict[str, np.ndarray]) -> None:
        """Текстовая визуализация матриц схожести"""
        filenames = similarity_matrices["filenames"]
        combined_matrix = similarity_matrices["combined"]

        print("\n📊 Матрица схожести (текстовый вид):")
        print("-" * (len(filenames) * 10 + 10))

        # Заголовок
        header = " " * 15
        for name in filenames:
            header += f"{name[:8]:>8} "
        print(header)
        print("-" * (len(filenames) * 10 + 10))

        # Данные
        for i, name in enumerate(filenames):
            row = f"{name[:12]:12} "
            for j in range(len(filenames)):
                row += f"{combined_matrix[i, j]:7.2f} "
            print(row)

    def _save_matrices_to_csv(self, similarity_matrices: Dict[str, np.ndarray]) -> None:
        """Сохранение матриц схожести в CSV файлы"""
        filenames = similarity_matrices["filenames"]

        for method in ["cosine", "lcs", "ngram", "combined"]:
            matrix = similarity_matrices[method]
            df = pd.DataFrame(matrix, index=filenames, columns=filenames)
            filename = f"{method}_similarity_matrix.csv"
            df.to_csv(filename)
            print(f"✓ Матрица {method} сохранена в {filename}")

    def generate_report(self, similarity_matrices: Dict[str, np.ndarray]) -> None:
        """
        Генерация отчета о потенциальном плагиате

        Args:
            similarity_matrices: словарь с матрицами схожести
        """
        if not similarity_matrices:
            print("❌ Нет данных для отчета")
            return

        combined_matrix = similarity_matrices["combined"]
        filenames = similarity_matrices["filenames"]
        n = len(filenames)

        print("\n" + "=" * 60)
        print("📊 ОТЧЕТ О ПОТЕНЦИАЛЬНОМ ПЛАГИАТЕ")
        print("=" * 60)

        potential_plagiarism = []

        for i in range(n):
            for j in range(i + 1, n):
                similarity = combined_matrix[i, j]
                if similarity >= self.min_threshold:
                    potential_plagiarism.append(
                        (
                            filenames[i],
                            filenames[j],
                            similarity,
                            similarity_matrices["cosine"][i, j],
                            similarity_matrices["lcs"][i, j],
                            similarity_matrices["ngram"][i, j],
                        )
                    )

        if potential_plagiarism:
            print(
                f"\n⚠️  Найдено {len(potential_plagiarism)} потенциальных случаев плагиата:"
            )
            print("-" * 100)
            print(
                f"{'Документ 1':<25} {'Документ 2':<25} {'Общая':<8} {'Cosine':<8} {'LCS':<8} {'N-gram':<8}"
            )
            print("-" * 100)

            for doc1, doc2, combined, cosine, lcs, ngram in sorted(
                potential_plagiarism, key=lambda x: x[2], reverse=True
            ):
                print(
                    f"{doc1:<25} {doc2:<25} {combined:.2%}    {cosine:.2%}    {lcs:.2%}    {ngram:.2%}"
                )
        else:
            print(
                f"\n✅ Потенциальных случаев плагиата не обнаружено (порог: {self.min_threshold:.0%})"
            )

        # Статистика
        print("\n" + "=" * 60)
        print("📈 СТАТИСТИКА АНАЛИЗА")
        print("=" * 60)
        print(f"Всего документов: {n}")
        print(f"Порог схожести: {self.min_threshold:.0%}")

        if n > 1:
            avg_similarity = np.mean(combined_matrix[np.triu_indices(n, k=1)])
            max_similarity = np.max(combined_matrix[np.triu_indices(n, k=1)])
            print(f"Средняя схожесть: {avg_similarity:.2%}")
            print(f"Максимальная схожесть: {max_similarity:.2%}")

            # Самые похожие документы
            if n > 1 and max_similarity > 0:
                print(f"\n🔍 Самые похожие документы:")
                indices = np.where(combined_matrix == max_similarity)
                for i, j in zip(indices[0], indices[1]):
                    if i < j:
                        print(f"  {filenames[i]} ↔ {filenames[j]}: {max_similarity:.2%}")

    def run_analysis(self, folder_path: str) -> Dict[str, np.ndarray]:
        """
        Полный запуск анализа плагиата

        Args:
            folder_path: путь к папке с документами

        Returns:
            Словарь с матрицами схожести
        """
        print("🚀 Запуск Educational Plagiarism Detector")
        print("=" * 60)

        try:
            # Загрузка документов
            self.load_documents(folder_path)

            if not self.documents:
                print("❌ Документы не найдены в указанной папке!")
                print(f"Поддерживаемые форматы: .txt, .pdf")
                print(f"Проверьте путь: {folder_path}")
                return {}

            print(f"\n📄 Загружено документов: {len(self.documents)}")

            # Обработка документов
            self.process_all_documents()

            # Расчет схожести
            similarity_matrices = self.calculate_similarity_matrix()

            if similarity_matrices:
                # Визуализация
                self.visualize_results(similarity_matrices)

                # Отчет
                self.generate_report(similarity_matrices)

            return similarity_matrices

        except Exception as e:
            print(f"\n❌ Критическая ошибка: {str(e)}")
            import traceback

            traceback.print_exc()
            return {}


def create_test_documents(folder_name: str = "test_documents"):
    """Создание тестовых документов для демонстрации"""
    import os

    os.makedirs(folder_name, exist_ok=True)

    # Тестовые тексты на русском языке (чтобы избежать проблем с NLTK)
    texts = {
        "документ1.txt": """Искусственный интеллект (ИИ) - это интеллект, демонстрируемый машинами, в отличие от естественного интеллекта, проявляемого людьми и животными. Ведущие учебники по ИИ определяют эту область как изучение интеллектуальных агентов: любой системы, которая воспринимает свое окружение и предпринимает действия, максимизирующие ее шансы на достижение целей.""",
        "документ2.txt": """Искусственный интеллект представляет собой интеллект, проявляемый машинами, в отличие от естественного интеллекта, демонстрируемого людьми и животными. Основные учебники по искусственному интеллекту определяют эту область как исследование интеллектуальных агентов: систем, которые воспринимают окружающую среду и действуют для максимизации вероятности достижения своих целей.""",
        "документ3.txt": """Машинное обучение - это раздел искусственного интеллекта, который фокусируется на разработке алгоритмов и статистических моделей, позволяющих компьютерным системам выполнять задачи без явных инструкций. Вместо этого они полагаются на выявление закономерностей и умозаключения.""",
        "документ4.txt": """Информатика - это изучение алгоритмических процессов, вычислительных машин и самого вычисления. Как дисциплина, информатика охватывает широкий круг тем - от теоретических исследований алгоритмов до практических вопросов реализации вычислительных систем в аппаратном и программном обеспечении.""",
    }

    # Создание файлов
    created_files = []
    for filename, content in texts.items():
        filepath = os.path.join(folder_name, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        created_files.append(filename)

    print(f"✓ Создано {len(created_files)} тестовых документов в папке '{folder_name}'")
    return folder_name


def create_test_documents_english(folder_name: str = "english_documents"):
    """Создание тестовых документов на английском языке"""
    import os

    os.makedirs(folder_name, exist_ok=True)

    # Тестовые тексты на английском языке
    texts = {
        "document1.txt": """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. Leading AI textbooks define the field as the study of intelligent agents: any system that perceives its environment and takes actions that maximize its chance of achieving its goals.""",
        "document2.txt": """Artificial intelligence is intelligence exhibited by machines, unlike the natural intelligence shown by humans and animals. Major AI textbooks describe this field as the research of intelligent agents: systems that perceive their surroundings and act to maximize the likelihood of accomplishing their objectives.""",
        "document3.txt": """Machine learning is a branch of artificial intelligence that focuses on the development of algorithms and statistical models that enable computer systems to perform tasks without explicit instructions. Instead, they rely on patterns and inference.""",
        "document4.txt": """Computer science is the study of algorithmic processes, computational machines, and computation itself. As a discipline, computer science spans a range of topics from theoretical studies of algorithms to the practical issues of implementing computational systems in hardware and software.""",
    }

    # Создание файлов
    created_files = []
    for filename, content in texts.items():
        filepath = os.path.join(folder_name, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        created_files.append(filename)

    print(f"✓ Created {len(created_files)} test documents in '{folder_name}' folder")
    return folder_name


# Пример использования
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Educational Plagiarism Detector - Демонстрация")
    print("=" * 60)

    print("\nВыберите язык документов:")
    print("1. Русский (рекомендуется - меньше зависимостей)")
    print("2. Английский")

    choice = input("Введите 1 или 2: ").strip()

    if choice == "1":
        print("\n🇷🇺 Используется русский язык")
        test_folder = create_test_documents()
        detector = PlagiarismDetector(min_similarity_threshold=0.4, language="russian")
    else:
        print("\n🇬🇧 Using English language")
        test_folder = create_test_documents_english()
        detector = PlagiarismDetector(min_similarity_threshold=0.4, language="english")

    # Запуск анализатора
    results = detector.run_analysis(test_folder)

    if results:
        print("\n" + "=" * 60)
        print("✅ Анализ успешно завершен!")
        print("=" * 60)
